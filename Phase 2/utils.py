# All imports
# Math
import math
import random
import cv2
import numpy as np
from scipy.stats import pearsonr
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# from sklearn.cluster import KMeans

# Torch
import torch
import torchvision.transforms as transforms
from torchvision.datasets import Caltech101
from torchvision.models import resnet50, ResNet50_Weights

# OS and env
import json
from os import getenv
from dotenv import load_dotenv
import warnings

load_dotenv()

# MongoDB
from pymongo import MongoClient

# Visualizing
import matplotlib.pyplot as plt


def getCollection(db, collection):
    """Load feature descriptor collection from MongoDB"""
    client = MongoClient("mongodb://localhost:27017")
    return client[db][collection]


def datasetTransform(image):
    """Transform while loading dataset as scaled tensors of shape (channels, (img_shape))"""
    return transforms.Compose(
        [
            transforms.ToTensor()  # ToTensor by default scales to [0,1] range, the input range for ResNet
        ]
    )(image)


def loadDataset(dataset):
    """Load TorchVision dataset with the defined transform"""
    return dataset(
        root=getenv("DATASET_PATH"),
        download=False,  # True if you wish to download for first time
        transform=datasetTransform,
    )


dataset = loadDataset(Caltech101)


class GridPartition:
    """Class transform to partition image into (rows, cols) grid"""

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

    def __call__(self, img):
        # img is in (C,(H,W)) format, so first element is channel
        img_width, img_height = img.size()[1:]
        cell_width = img_width // self.cols
        cell_height = img_height // self.rows

        grids = []
        for i in range(self.rows):
            for j in range(self.cols):
                left = j * cell_width
                right = left + cell_width

                top = i * cell_height
                bottom = top + cell_height

                # Slice out
                grid = img[:, left:right, top:bottom]
                grids.append(grid)

        return grids


def compute_color_moments(grid_cell):
    """Compute color moments (mean, std. deviation, skewness), assuming RGB channels"""
    grid_cell = np.array(grid_cell)  # Convert tensor to NumPy array
    moments = []

    for channel in range(3):  # Iterate over RGB channels
        channel_data = grid_cell[:, :, channel]
        mean = np.mean(channel_data)
        std_dev = np.std(channel_data)

        # Avoiding NaN values
        skew_cubed = np.mean((channel_data - mean) ** 3)
        if skew_cubed > 0:
            skew = math.pow(skew_cubed, float(1) / 3)
        elif skew_cubed < 0:
            skew = -math.pow(abs(skew_cubed), float(1) / 3)
        else:
            skew = 0

        moments.append([mean, std_dev, skew])

    return moments


def compute_color_moments_for_grid(grid):
    color_moments = [compute_color_moments(grid_cell) for grid_cell in grid]
    return np.array(color_moments).flatten()


def combine_color_moments(grid_color_moments):
    return torch.Tensor(grid_color_moments).view(
        10, 10, 3, 3
    )  # resize as needed: 10x10 grid, 3 channels per cell, 3 moments per channel


# Transform pipeline to get CM10x10 900-dimensional feature descriptor
CM_transform = transforms.Compose(
    [
        transforms.Resize((100, 300)),  # resize to H:W=100:300
        GridPartition(
            rows=10, cols=10
        ),  # partition into grid of 10 rows, 10 columns as a list
        compute_color_moments_for_grid,
        combine_color_moments,
    ]
)


def compute_gradient_histogram(grid_cell):
    """Compute HOG using [-1,0,1] masks for gradient"""
    histograms = []

    # Convert grid cell to NumPy array
    grid_array = np.array(grid_cell, dtype=np.float32)
    grid_array = grid_array.reshape(
        grid_array.shape[1], grid_array.shape[2]
    )  # ignore extra dimension

    # Compute the gradient using first-order central differences
    dx = cv2.Sobel(
        grid_array, cv2.CV_32F, dx=1, dy=0, ksize=1
    )  # first order x derivative = [-1, 0, 1]
    dy = cv2.Sobel(
        grid_array, cv2.CV_32F, dx=0, dy=1, ksize=1
    )  # first order y derivative = [-1, 0, 1]^T

    # Compute magnitude and direction of gradients
    magnitude = np.sqrt(dx**2 + dy**2)
    direction = np.arctan2(dy, dx) * 180 / np.pi  # in degrees

    # Compute HOG - 9 bins, counted across the range of -180 to 180 degrees, weighted by gradient magnitude
    histogram, _ = np.histogram(direction, bins=9, range=(-180, 180), weights=magnitude)

    histograms.append(histogram)

    return histograms


def compute_histograms_for_grid(grid):
    histograms = [compute_gradient_histogram(grid_cell) for grid_cell in grid]
    return np.array(histograms).flatten()


def combine_histograms(grid_histograms):
    return torch.Tensor(grid_histograms).view(10, 10, 9)


# Transform pipeline to get HOG10x10 900-dimensional feature descriptor
HOG_transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),  # grayscale transform
        transforms.Resize((100, 300)),  # resize to H:W=100:300
        GridPartition(
            rows=10, cols=10
        ),  # partition into grid of 10 rows, 10 columns as a list
        compute_histograms_for_grid,
        combine_histograms,
    ]
)


def loadResnet():
    """Load ResNet50 pre-trained model with default weights"""
    # Load model
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    # try to use Nvidia GPU
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        torch.cuda.empty_cache()
    else:
        dev = torch.device("cpu")

    model = model.to(dev)
    model.eval()  # switch to inference mode - important! since we're using pre-trained model
    return model, dev


model, dev = loadResnet()


class FeatureExtractor(torch.nn.Module):
    """Feature extractor module for all layers at once"""

    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: None for layer in layers}  # store layer outputs here

        # Create hooks for all specified layers at once
        for layer_id in layers:
            layer = dict(self.model.named_modules())[
                layer_id
            ]  # get actual layer in the model
            layer.register_forward_hook(
                self.save_outputs_hook(layer_id)
            )  # register feature extractor hook on layer

    # Hook to save output of layer
    def save_outputs_hook(self, layer_id):
        def fn(_module, _input, output):
            self._features[layer_id] = output

        return fn

    # Forward pass returns extracted features
    def forward(self, input):
        _ = self.model(input)
        return self._features


def resnet_extractor(image):
    """Extract image features from avgpool, layer3 and fc layers of ResNet50"""
    resized_image = (
        torch.Tensor(np.array(transforms.Resize((224, 224))(image)).flatten())
        .view(1, 3, 224, 224)
        .to(dev)
    )

    # Attach all hooks on model and extract features
    resnet_features = FeatureExtractor(model=model, layers=["avgpool", "layer3", "fc"])
    features = resnet_features(resized_image)

    avgpool_2048 = features["avgpool"]
    # Reshape the vector into row pairs of elements and average across rows
    avgpool_1024_fd = torch.mean(avgpool_2048.view(-1, 2), axis=1)

    layer3_1024_14_14 = features["layer3"]
    # Reshape the vector into 1024 rows of 196 elements and average across rows
    layer3_1024_fd = torch.mean(layer3_1024_14_14.view(1024, -1), axis=1)

    fc_1000_fd = features["fc"].view(1000)

    return (
        avgpool_1024_fd.detach().cpu().tolist(),
        layer3_1024_fd.detach().cpu().tolist(),
        fc_1000_fd.detach().cpu().tolist(),
    )


def get_all_fd(image_id, img=None, label=None):
    """Get all feature descriptors of a given image"""
    img_shape = np.array(img).shape
    if img_shape[0] >= 3:
        true_channels = 3
    else:
        # stacking the grayscale channel on itself thrice to get RGB dimensions
        img = torch.tensor(np.stack((np.array(img[0, :, :]),) * 3, axis=0))
        true_channels = 1

    cm_fd = CM_transform(img).tolist()
    hog_fd = HOG_transform(img).tolist()
    avgpool_1024_fd, layer3_1024_fd, fc_1000_fd = resnet_extractor(img)

    return {
        "image_id": image_id,
        "true_label": label,
        "true_channels": true_channels,
        "cm_fd": cm_fd,
        "hog_fd": hog_fd,
        "avgpool_fd": avgpool_1024_fd,
        "layer3_fd": layer3_1024_fd,
        "fc_fd": fc_1000_fd,
    }


def euclidean_distance_measure(img_1_fd, img_2_fd):
    img_1_fd_reshaped = img_1_fd.flatten()
    img_2_fd_reshaped = img_2_fd.flatten()

    # Calculate Euclidean distance
    return math.dist(img_1_fd_reshaped, img_2_fd_reshaped)


def cosine_distance_measure(img_1_fd, img_2_fd):
    img_1_fd_reshaped = img_1_fd.flatten()
    img_2_fd_reshaped = img_2_fd.flatten()

    # Calculate dot product
    dot_product = np.dot(img_1_fd_reshaped, img_2_fd_reshaped.T)

    # Calculate magnitude (L2 norm) of the feature descriptor
    magnitude1 = np.linalg.norm(img_1_fd_reshaped)
    magnitude2 = np.linalg.norm(img_2_fd_reshaped)

    # Calculate cosine distance (similarity is higher => distance should be lower, so subtract from 1)
    cosine_similarity = dot_product / (magnitude1 * magnitude2)
    return 1 - cosine_similarity


def pearson_distance_measure(img_1_fd, img_2_fd):
    # Replace nan with 0 (color moments)
    img_1_fd_reshaped = img_1_fd.flatten()
    img_2_fd_reshaped = img_2_fd.flatten()

    # Invert and scale in half to fit the actual range [-1, 1] into the new range [0, 1]
    # such that lower distance implies more similarity
    return 0.5 * (1 - pearsonr(img_1_fd_reshaped, img_2_fd_reshaped).statistic)


valid_feature_models = {
    "cm": "cm_fd",
    "hog": "hog_fd",
    "avgpool": "avgpool_fd",
    "layer3": "layer3_fd",
    "fc": "fc_fd",
}
valid_distance_measures = {
    "euclidean": euclidean_distance_measure,
    "cosine": cosine_distance_measure,
    "pearson": pearson_distance_measure,
}
feature_distance_matches = {
    "cm_fd": euclidean_distance_measure,
    "hog_fd": cosine_distance_measure,
    "layer3_fd": pearson_distance_measure,
    "avgpool_fd": pearson_distance_measure,
    "fc_fd": pearson_distance_measure,
}


def show_similar_images_for_image(
    fd_collection,
    target_image_id,
    target_image=None,
    target_label=None,
    k=10,
    feature_model="fc",
    distance_measure=pearson_distance_measure,
    save_plots=False,
):
    """Set `target_image_id = -1` if giving image data and label manually"""

    assert (
        feature_model in valid_feature_models.values()
    ), "feature_model should be one of " + str(list(valid_feature_models.keys()))

    assert (
        distance_measure in valid_distance_measures.values()
    ), "distance_measure should be one of " + str(list(valid_distance_measures.keys()))

    all_images = fd_collection.find()

    # if target from dataset
    if target_image_id != -1:
        print(
            "Showing {} similar images for image ID {}, using {} for {} feature descriptor...".format(
                k, target_image_id, distance_measure.__name__, feature_model
            )
        )

        # store distance to target_image itself
        min_dists = {target_image_id: 0}

        # in phase 2, we only have even-numbered image IDs in database
        if target_image_id % 2 == 0:
            # Get target image's feature descriptors from database
            target_image_fds = fd_collection.find_one({"image_id": target_image_id})
        else:
            # Calculate target image's feature descriptors
            target_image, target_label = dataset[target_image_id]
            target_image_fds = get_all_fd(target_image_id, target_image, target_label)

        target_image_fd = np.array(target_image_fds[feature_model])

        for cur_img in all_images:
            cur_img_id = cur_img["image_id"]
            # skip target itself
            if cur_img_id == target_image_id:
                continue
            cur_img_fd = np.array(cur_img[feature_model])

            cur_dist = distance_measure(
                cur_img_fd,
                target_image_fd,
            )

            # store first k images irrespective of distance (so that we store no more than k minimum distances)
            if len(min_dists) < k + 1:
                min_dists[cur_img_id] = cur_dist

            # if lower distance:
            elif cur_dist < max(min_dists.values()):
                # add to min_dists
                min_dists.update({cur_img_id: cur_dist})
                # remove greatest distance by index
                min_dists.pop(max(min_dists, key=min_dists.get))

        min_dists = dict(sorted(min_dists.items(), key=lambda item: item[1]))

        # Display the target image along with the k images
        fig, axs = plt.subplots(1, k + 1, figsize=(48, 12))
        for idx, (img_id, distance) in enumerate(min_dists.items()):
            cur_img, _cur_label = dataset[img_id]
            axs[idx].imshow(transforms.ToPILImage()(cur_img))
            if idx == 0:
                axs[idx].set_title(f"Target image")
            else:
                axs[idx].set_title(f"Distance: {round(distance, 3)}")
            axs[idx].axis("off")

        if save_plots:
            plt.savefig(
                f"Plots/Image_{target_image_id}_{feature_model}_{distance_measure.__name__}_k{k}.png"
            )
        plt.show()

    # else, if target from some image file
    else:
        print(
            "Showing {} similar images for given image, using {} for {} feature descriptor...".format(
                k, distance_measure.__name__, feature_model
            )
        )

        # store distance to target_image itself
        min_dists = {-1: 0}

        target_image_fds = get_all_fd(-1, target_image, target_label)
        target_image_fd = np.array(target_image_fds[feature_model])

        for cur_img in all_images:
            cur_img_id = cur_img["image_id"]
            cur_img_fd = np.array(cur_img[feature_model])
            cur_dist = distance_measure(
                cur_img_fd,
                target_image_fd,
            )

            # store first k images irrespective of distance (so that we store no more than k minimum distances)
            if len(min_dists) < k + 1:
                min_dists[cur_img_id] = cur_dist

            # if lower distance:
            elif cur_dist < max(min_dists.values()):
                # add to min_dists
                min_dists.update({cur_img_id: cur_dist})
                # remove greatest distance by index
                min_dists.pop(max(min_dists, key=min_dists.get))

        min_dists = dict(sorted(min_dists.items(), key=lambda item: item[1]))

        # Display the target image along with the k images
        fig, axs = plt.subplots(1, k + 1, figsize=(48, 12))
        for idx, (img_id, distance) in enumerate(min_dists.items()):
            if idx == 0:
                axs[idx].imshow(transforms.ToPILImage()(target_image))
                axs[idx].set_title(f"Target image")
            else:
                cur_img, _cur_label = dataset[img_id]
                axs[idx].imshow(transforms.ToPILImage()(cur_img))
                axs[idx].set_title(f"Distance: {round(distance, 3)}")
            axs[idx].axis("off")

        if save_plots:
            plt.savefig(
                f"Plots/Image_{target_image_id}_{feature_model}_{distance_measure.__name__}_k{k}.png"
            )
        plt.show()


def calculate_label_representatives(fd_collection, label, feature_model):
    """Calculate representative feature vector of a label as the mean of all feature vectors under a feature model"""

    label_fds = [
        img_fds[feature_model]  # get the specific feature model's feature vector
        for img_fds in fd_collection.find(
            {"true_label": label}
        )  # repeat for all images
    ]

    # Calculate mean across each dimension
    # and build a mean vector out of these means
    label_mean_vector = [sum(col) / len(col) for col in zip(*label_fds)]
    return label_mean_vector


def show_similar_images_for_label(
    fd_collection,
    target_label,
    k=10,
    feature_model="fc",
    distance_measure=pearson_distance_measure,
    save_plots=False,
):
    assert (
        feature_model in valid_feature_models.values()
    ), "feature_model should be one of " + str(list(valid_feature_models.keys()))

    assert (
        distance_measure in valid_distance_measures.values()
    ), "distance_measure should be one of " + str(list(valid_distance_measures.keys()))

    all_images = fd_collection.find()

    print(
        "Showing {} similar images for label {}, using {} for {} feature descriptor...".format(
            k, target_label, distance_measure.__name__, feature_model
        )
    )

    # store distance to target_label itself ({image_id: distance}, -1 for target label)
    min_dists = {}

    # Calculate representative feature vector for label
    label_rep = calculate_label_representatives(
        fd_collection, target_label, feature_model
    )

    for cur_img in all_images:
        cur_img_id = cur_img["image_id"]
        cur_img_fd = np.array(cur_img[feature_model])

        cur_dist = distance_measure(
            cur_img_fd,
            np.array(label_rep),
        )

        # store first k images irrespective of distance (so that we store no more than k minimum distances)
        if len(min_dists) < k:
            min_dists[cur_img_id] = cur_dist

        # if lower distance:
        elif cur_dist < max(min_dists.values()):
            # add to min_dists
            min_dists.update({cur_img_id: cur_dist})
            # remove greatest distance by index
            min_dists.pop(max(min_dists, key=min_dists.get))

    min_dists = dict(sorted(min_dists.items(), key=lambda item: item[1]))

    # Display the k images
    fig, axs = plt.subplots(1, k, figsize=(48, 12))
    for idx, (img_id, distance) in enumerate(min_dists.items()):
        cur_img, _cur_label = dataset[img_id]
        axs[idx].imshow(transforms.ToPILImage()(cur_img))
        axs[idx].set_title(f"Distance: {round(distance, 3)}")
        axs[idx].axis("off")

    if save_plots:
        plt.savefig(
            f"Plots/Label_{target_label}_{feature_model}_{distance_measure.__name__}_k{k}.png"
        )
    plt.show()


valid_dim_reduction_methods = {
    "svd": 1,
    "nmf": 2,
    "lda": 3,
    "kmeans": 4,
}


class KMeans:
    def __init__(self, n_clusters, tol=0.001, max_iter=300, verbose=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_ = {}
        self.verbose = verbose

    def fit(self, data):
        """Iterative fitting clusters on data of `(n_samples,n_features)` dimensions"""

        # Randomly select centroid start points with uniform distribution from dataset
        min_, max_ = np.min(data, axis=0), np.max(data, axis=0)
        self.cluster_centers_ = {
            i: np.random.uniform(min_, max_) for i in range(self.n_clusters)
        }

        if self.verbose > 0:
            print("Initialized centroids")
        for itr in range(self.max_iter):
            print(f"Iteration {itr}")
            self.clusters = {}

            for j in range(self.n_clusters):
                self.clusters[j] = []

            for feature_set in data:
                # TODO: Should this be modified to use different distance measures
                # based on the feature set?
                distances = [
                    np.linalg.norm(feature_set - self.cluster_centers_[i])
                    for i in range(len(self.cluster_centers_))
                ]

                # Put data point into closest cluster
                cluster = np.argmin(distances)
                self.clusters[cluster].append(feature_set)

            prev_centroids = self.cluster_centers_

            for c in self.cluster_centers_:
                if isinstance(self.cluster_centers_[c], np.ndarray):
                    if np.isnan(self.cluster_centers_[c]).any():
                        # Reinitialize centroid to a random point in the dataset
                        self.cluster_centers_[c] = np.random.uniform(min_, max_)
                    else:
                        # Compute the mean of non-empty cluster
                        self.cluster_centers_[c] = np.mean(self.clusters[c], axis=0)
                elif np.isnan(self.cluster_centers_[c]):
                    # Reinitialize centroid to a random point in the dataset
                    self.cluster_centers_[c] = np.random.uniform(min_, max_)

            # Check if centroids have converged
            optimized = True
            for c in self.cluster_centers_:
                prev_centroid = prev_centroids[c]
                current_centroid = self.cluster_centers_[c]
                convergence_tol = np.sum(abs(
                    (prev_centroid - current_centroid) / prev_centroid * 100.0
                ))
                if convergence_tol > self.tol:
                    optimized = False
                    if self.verbose > 0:
                        print(f"Iter {itr} - Not converged yet")
                    break

            if itr > 10 and optimized:
                if self.verbose > 0:
                    print(f"Iter {itr} - Converged")
                break

        return self

    def transform(self, data):
        """Transform data of `(n_samples,n_features)` dimensions to `(n_samples,n_clusters)` using fitted model"""

        Y = np.empty((len(data), self.n_clusters))

        for idx, feature_set in enumerate(data):
            # TODO: Could this be modified to use different distance measures
            # based on the feature set?
            Y[idx] = np.array(
                [
                    np.linalg.norm(feature_set - self.cluster_centers_[i])
                    for i in range(len(self.cluster_centers_))
                ]
            )

        return Y


def extract_latent_semantics(
    fd_collection, k, feature_model, dim_reduction_method, top_images=None
):
    """
    Extract latent semantics for entire collection at once for a given feature_model and dim_reduction_method, and display the imageID-semantic weight pairs

    Leave `top_images` blank to display all imageID-weight pairs
    """

    assert (
        feature_model in valid_feature_models.values()
    ), "feature_model should be one of " + str(list(valid_feature_models.keys()))
    assert (
        dim_reduction_method in valid_dim_reduction_methods.keys()
    ), "dim_reduction_method should be one of " + str(
        list(valid_dim_reduction_methods.keys())
    )

    all_images = list(fd_collection.find())
    feature_vectors = np.array([img[feature_model] for img in all_images])
    feature_labels = [img["true_label"] for img in all_images]
    feature_ids = [img["image_id"] for img in all_images]

    top_img_str = ""
    if top_images is not None:
        top_img_str = f" (showing only top {top_images} image-weight pairs for each latent semantic)"
    print(
        "Applying {} on the {} space to get {} latent semantics{}...".format(
            dim_reduction_method, feature_model, k, top_img_str
        )
    )

    displayed_latent_semantics = {}
    all_latent_semantics = {}

    match valid_dim_reduction_methods[dim_reduction_method]:
        # singular value decomposition
        # sparse version of SVD to get only k singular values
        case 1:
            U, S, V_T = svds(feature_vectors, k=k)

            all_latent_semantics = {
                "image-semantic": U.tolist(),
                "semantics-core": S.tolist(),
                "semantic-feature": V_T.tolist(),
            }

            # for each latent semantic, sort imageID-weight pairs by weights in descending order
            displayed_latent_semantics = [
                sorted(
                    list(zip(feature_ids, latent_semantic)),
                    key=lambda x: x[1],
                    reverse=True,
                )[:top_images]
                for latent_semantic in U.T
            ]

        # non-negative matrix factorization
        case 2:
            # NNMF requires non-negative input data
            # so shift the input by subtracting the smallest value
            min_value = np.min(feature_vectors)
            feature_vectors_shifted = feature_vectors - min_value

            model = NMF(
                n_components=k,
                init="random",
                solver="cd",
                alpha_H=0.01,
                alpha_W=0.01,
                max_iter=10000,
            )
            model.fit(feature_vectors_shifted)

            W = model.transform(feature_vectors_shifted)
            H = model.components_

            all_latent_semantics = {
                "image-semantic": W.tolist(),
                "semantic-feature": H.tolist(),
            }

            # for each latent semantic, sort imageID-weight pairs by weights in descending order
            displayed_latent_semantics = [
                sorted(
                    list(zip(feature_ids, latent_semantic)),
                    key=lambda x: x[1],
                    reverse=True,
                )[:top_images]
                for latent_semantic in W.T
            ]

        # unsupervised LDA to extract topics (Latent Dirichlet Allocation)
        # Note: LDA takes a bit of time
        case 3:
            # LDA requires non-negative input data
            # so shift the input by subtracting the smallest value
            min_value = np.min(feature_vectors)
            feature_vectors_shifted = feature_vectors - min_value

            model = LatentDirichletAllocation(
                n_components=k, learning_method="online", verbose=4
            )
            model.fit(feature_vectors_shifted)

            # K (k x fd_dim) is the factor matrix for latent semantic-feature pairs
            K = model.components_
            # X (4339 x k) is the other factor matrix for image ID-latent semantic pairs
            X = model.transform(feature_vectors_shifted)

            all_latent_semantics = {
                "image-semantic": X.tolist(),
                "semantic-feature": K.tolist(),
            }

            # for each latent semantic, sort imageID-weight pairs by weights in descending order
            displayed_latent_semantics = [
                sorted(
                    list(zip(feature_ids, latent_semantic)),
                    key=lambda x: x[1],
                    reverse=True,
                )[:top_images]
                for latent_semantic in X.T
            ]

        # k-means clustering to reduce to k clusters/dimensions
        case 4:
            model = KMeans(n_clusters=k, verbose=2).fit(feature_vectors)
            CC = model.cluster_centers_
            Y = model.transform(feature_vectors)

            all_latent_semantics = {
                "image-semantic": Y.tolist(),
                "semantic-feature": list(CC.values()),
            }

            # for each latent semantic, sort imageID-weight pairs by weights in descending order
            displayed_latent_semantics = [
                sorted(
                    list(zip(feature_ids, latent_semantic)),
                    key=lambda x: x[1],
                    reverse=False,
                )[:top_images]
                for latent_semantic in Y.T
            ]

    for idx, latent_semantic in enumerate(displayed_latent_semantics):
        print(f"Latent semantic no. {idx}")
        for image_id, weight in latent_semantic:
            print(f"Image_ID\t{image_id}\t-\tWeight\t{weight}")

    with open(
        f"{feature_model}-{dim_reduction_method}-{k}-semantics.json",
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(all_latent_semantics, output_file, ensure_ascii=False)
