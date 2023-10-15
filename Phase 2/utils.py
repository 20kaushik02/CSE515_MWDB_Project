# All imports
# Math
import math
import random
import cv2
import numpy as np
from scipy.stats import pearsonr
# from scipy.sparse.linalg import svds
# from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation

# from sklearn.cluster import KMeans

# Torch
import torch
import torchvision.transforms as transforms
from torchvision.datasets import Caltech101
from torchvision.models import resnet50, ResNet50_Weights

import tensorly as tl

# OS and env
import json
from os import getenv
from dotenv import load_dotenv
import warnings
from joblib import dump, load

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
NUM_LABELS = 101
NUM_IMAGES = 4339


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


def resnet_output(image):
    """Get image features from ResNet50 (full execution) and apply a softmax layer"""
    resized_image = (
        torch.Tensor(np.array(transforms.Resize((224, 224))(image)).flatten())
        .view(1, 3, 224, 224)
        .to(dev)
    )

    with torch.no_grad():
        features = model(resized_image)
        features = torch.nn.Softmax()(features)

    return features.detach().cpu().tolist()


def get_all_fd(image_id, given_image=None, given_label=None):
    """Get all feature descriptors of a given image"""
    if image_id == -1:
        img, label = given_image, given_label
    else:
        img, label = dataset[image_id]
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
    resnet_fd = resnet_output(img)

    return {
        "image_id": image_id,
        "true_label": label,
        "true_channels": true_channels,
        "cm_fd": cm_fd,
        "hog_fd": hog_fd,
        "avgpool_fd": avgpool_1024_fd,
        "layer3_fd": layer3_1024_fd,
        "fc_fd": fc_1000_fd,
        "resnet_fd": resnet_fd,
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
    "resnet": "resnet_fd",
}
valid_latent_spaces = {
    "ls1": "",
    "ls2": "cp",
    "ls3": "label_sim",
    "ls4": "image_sim",
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
    "resnet_fd": pearson_distance_measure,
}


def show_similar_images_for_image(
    fd_collection,
    target_image_id,
    target_image=None,
    target_label=None,
    k=10,
    feature_model="fc_fd",
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
                f"Plots/Image_{target_image_id}_{feature_model}_{distance_measure.__name__}_{k}_images.png"
            )
        plt.show()


def calculate_label_representatives(fd_collection, label, feature_model):
    """Calculate representative feature vector of a label as the mean of all feature vectors under a feature model"""

    label_fds = [
        np.array(
            img_fds[feature_model]
        ).flatten()  # get the specific feature model's feature vector
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
    feature_model="fc_fd",
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
        cur_img_fd = np.array(cur_img[feature_model]).flatten()

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
            f"Plots/Label_{target_label}_{feature_model}_{distance_measure.__name__}_{k}_images.png"
        )
    plt.show()


def show_similar_labels_for_image(
    fd_collection,
    target_image_id,
    target_image=None,
    target_label=None,
    k=10,
    feature_model="fc_fd",
    distance_measure=pearson_distance_measure,
    save_plots=False,
):
    assert (
        feature_model in valid_feature_models.values()
    ), "feature_model should be one of " + str(valid_feature_models.keys())

    assert (
        distance_measure in valid_distance_measures.values()
    ), "distance_measure should be one of " + str(list(valid_distance_measures.keys()))

    # if target from dataset
    if target_image_id != -1:
        print(
            "Showing {} similar labels for image ID {}, using {} for {} feature descriptor...".format(
                k, target_image_id, distance_measure.__name__, feature_model
            )
        )

        # store target_image itself
        min_dists = {target_image_id: 0}

        if target_image_id % 2 == 0:
            # Get target image's feature descriptors from database
            target_image = fd_collection.find_one({"image_id": target_image_id})
        else:
            # Calculate target image's feature descriptors
            target_image = get_all_fd(target_image_id)

        target_image_fd = np.array(target_image[feature_model])
        target_label = target_image["true_label"]

    else:
        print(
            "Showing {} similar labels for given image, using {} for {} feature descriptor...".format(
                k, distance_measure.__name__, feature_model
            )
        )

        # store distance to target_image itself
        min_dists = {-1: 0}

        target_image_fds = get_all_fd(-1, target_image, target_label)
        target_image_fd = np.array(target_image_fds[feature_model])

    label_dict = {target_image_id: target_label}

    all_images = fd_collection.find({})
    for cur_img in all_images:
        cur_img_id = cur_img["image_id"]
        # skip target itself
        if cur_img_id == target_image_id:
            continue
        cur_img_fd = np.array(cur_img[feature_model]).flatten()
        cur_dist = distance_measure(
            cur_img_fd,
            target_image_fd,
        )
        cur_label = cur_img["true_label"]

        # store first k images irrespective of distance (so that we store no more than k minimum distances)
        if len(min_dists) < k + 1 and cur_label not in label_dict.values():
            min_dists[cur_img_id] = cur_dist
            label_dict[cur_img_id] = cur_label

        # if lower distance:
        elif (
            cur_dist < max(min_dists.values()) and cur_label not in label_dict.values()
        ):
            # add to min_dists
            min_dists.update({cur_img_id: cur_dist})
            label_dict.update({cur_img_id: cur_label})
            # remove label with greatest distance by index
            pop_key = max(min_dists, key=min_dists.get)
            min_dists.pop(pop_key)
            label_dict.pop(pop_key)

    min_dists = dict(sorted(min_dists.items(), key=lambda item: item[1]))

    fig, axs = plt.subplots(1, k, figsize=(48, 12))
    for idx, image_id in enumerate(min_dists.keys()):
        if image_id == target_image_id:
            continue
        else:
            sample_image, sample_label = dataset[image_id]
            axs[idx - 1].imshow(transforms.ToPILImage()(sample_image))
            axs[idx - 1].set_title(
                f"Label: {label_dict[image_id]}; Distance: {min_dists[image_id]}"
            )
        axs[idx - 1].axis("off")

    if save_plots:
        plt.savefig(
            f"Plots/Image_{target_image_id}_{feature_model}_{distance_measure.__name__}_{k}_labels.png"
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
        self.cluster_centers_ = None
        self.verbose = verbose

    def _initialize_centroids(self, data):
        random_indices = np.random.choice(data.shape[0], self.n_clusters, replace=False)
        self.cluster_centers_ = data[random_indices]

    def fit(self, data):
        data = np.array(data)
        self._initialize_centroids(data)

        if self.verbose > 0:
            print("Initialized centroids")

        for itr in range(self.max_iter):
            clusters = {j: [] for j in range(self.n_clusters)}

            for feature_set in data:
                distances = np.linalg.norm(feature_set - self.cluster_centers_, axis=1)
                cluster = np.argmin(distances)
                clusters[cluster].append(feature_set)

            prev_centroids = np.copy(self.cluster_centers_)

            for c in range(self.n_clusters):
                if len(clusters[c]) > 0:
                    self.cluster_centers_[c] = np.mean(clusters[c], axis=0)
                else:
                    # Reinitialize centroid to a random point in the dataset
                    random_index = np.random.choice(data.shape[0])
                    self.cluster_centers_[c] = data[random_index]

            # Check if centroids have converged
            convergence_tol = np.sum(
                np.abs((prev_centroids - self.cluster_centers_) / prev_centroids)
            )
            if convergence_tol < self.tol:
                if self.verbose > 0:
                    print(f"Iteration {itr} - Converged")
                break

        return self

    def transform(self, data):
        if self.cluster_centers_ is None:
            raise ValueError("Fit the model first using the 'fit' method.")

        data = np.array(data)
        Y = np.empty((data.shape[0], self.n_clusters))

        for idx, feature_set in enumerate(data):
            Y[idx] = np.linalg.norm(feature_set - self.cluster_centers_, axis=1)

        return Y


def svd(matrix, k):
    # Step 1: Compute the covariance matrix
    cov_matrix = np.dot(matrix.T, matrix)

    # Step 2: Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Step 3: Sort the eigenvalues and corresponding eigenvectors
    sort_indices = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[sort_indices]
    eigenvectors = eigenvectors[:, sort_indices]

    # Step 4: Compute the singular values and the left and right singular vectors
    singular_values = np.sqrt(eigenvalues)
    left_singular_vectors = np.dot(matrix, eigenvectors)
    right_singular_vectors = eigenvectors

    # Step 5: Normalize the singular vectors
    for i in range(left_singular_vectors.shape[1]):
        left_singular_vectors[:, i] /= singular_values[i]

    for i in range(right_singular_vectors.shape[1]):
        right_singular_vectors[:, i] /= singular_values[i]

    # Keep only the top k singular values and their corresponding vectors
    singular_values = singular_values[:k]
    left_singular_vectors = left_singular_vectors[:, :k]
    right_singular_vectors = right_singular_vectors[:, :k]

    return left_singular_vectors, np.diag(singular_values), right_singular_vectors.T


def nmf(matrix, k, H=None, update_H=True, num_iterations=100):
    """
    Non-negative matrix factorization by multiplicative update

    Pass `H` and `update_H=False` to transform given data as per the given H matrix, else leave `H=None` and `update_H=True` to fit and transform
    """
    d1, d2 = matrix.shape
    # Initialize W and H matrices with random non-negative values
    W = np.random.rand(d1, k)
    if update_H is True:
        H = np.random.rand(k, d2)

    for iteration in range(num_iterations):
        if update_H is True:
            # Update H matrix
            numerator_h = np.dot(W.T, matrix)
            denominator_h = np.dot(np.dot(W.T, W), H)
            H *= numerator_h / denominator_h

        # Update W matrix
        numerator_w = np.dot(matrix, H.T)
        denominator_w = np.dot(W, np.dot(H, H.T))
        W *= numerator_w / denominator_w

    return W, H


def extract_latent_semantics_from_feature_model(
    fd_collection,
    k,
    feature_model,
    dim_reduction_method,
    top_images=None,
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
    feature_ids = [img["image_id"] for img in all_images]

    top_img_str = ""
    if top_images is not None:
        top_img_str = f" (showing only top {top_images} image-weight pairs for each latent semantic)"

    feature_vectors = np.array(
        [np.array(img[feature_model]).flatten() for img in all_images]
    )
    print(
        "Applying {} on the {} space to get {} latent semantics{}...".format(
            dim_reduction_method, feature_model, k, top_img_str
        )
    )

    displayed_latent_semantics = {}
    all_latent_semantics = {}

    match valid_dim_reduction_methods[dim_reduction_method]:
        # singular value decomposition
        case 1:
            U, S, V_T = svd(feature_vectors, k=k)

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

            W, H = nmf(feature_vectors_shifted, k)

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

            # K (k x fd_dim) is the pseudocount for latent semantic-feature pairs
            K = model.components_
            # X (4339 x k) is the image-semantic distribution (image ID-latent semantic pairs)
            X = model.transform(feature_vectors_shifted)

            all_latent_semantics = {
                "image-semantic": X.tolist(),
                "semantic-feature": K.tolist(),
            }

            dump(model, f"{feature_model}-{dim_reduction_method}-{k}-model.joblib")

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
                "semantic-feature": CC.tolist(),
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

    if valid_dim_reduction_methods[dim_reduction_method] == 4:
        print("Note: for K-Means we display distances, in ascending order")
    for idx, latent_semantic in enumerate(displayed_latent_semantics):
        print(f"Latent semantic no. {idx}")
        for image_id, weight in latent_semantic:
            if valid_dim_reduction_methods[dim_reduction_method] == 4:
                print(f"Image_ID\t{image_id}\t-\tDistance\t{weight}")
            else:
                print(f"Image_ID\t{image_id}\t-\tWeight\t{weight}")

    with open(
        f"{feature_model}-{dim_reduction_method}-{k}-semantics.json",
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(all_latent_semantics, output_file, ensure_ascii=False)


def extract_latent_semantics_from_sim_matrix(
    sim_matrix,
    feature_model,
    sim_type,
    k,
    dim_reduction_method,
    top_images=None,
):
    """
    Extract latent semantics for a given similarity matrix for a given dim_reduction_method, and display the object-semantic weight pairs

    Leave `top_images` blank to display all imageID-weight pairs
    """

    assert sim_type in ["image", "label"], "sim_type should be one of " + str(
        ["image", "label"]
    )
    assert (
        feature_model in valid_feature_models.values()
    ), "feature_model should be one of " + str(list(valid_feature_models.keys()))
    assert (
        dim_reduction_method in valid_dim_reduction_methods.keys()
    ), "dim_reduction_method should be one of " + str(
        list(valid_dim_reduction_methods.keys())
    )
    assert len(sim_matrix) == len(sim_matrix[0]), "sim_matrix must be square matrix"

    top_img_str = ""
    if top_images is not None:
        top_img_str = f" (showing only top {top_images} {sim_type}-weight pairs for each latent semantic)"

    feature_vectors = sim_matrix
    feature_ids = list(range(len(sim_matrix)))

    print(
        "Applying {} on the given similarity matrix to get {} latent semantics{}...".format(
            dim_reduction_method, k, top_img_str
        )
    )

    displayed_latent_semantics = {}
    all_latent_semantics = {}

    match valid_dim_reduction_methods[dim_reduction_method]:
        # singular value decomposition
        case 1:
            U, S, V_T = svd(feature_vectors, k=k)

            all_latent_semantics = {
                "image-semantic": U.tolist(),
                "semantics-core": S.tolist(),
                "semantic-feature": V_T.tolist(),
            }

            # for each latent semantic, sort object-weight pairs by weights in descending order
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

            W, H = nmf(feature_vectors_shifted, k)

            all_latent_semantics = {
                "image-semantic": W.tolist(),
                "semantic-feature": H.tolist(),
            }

            # for each latent semantic, sort object-weight pairs by weights in descending order
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

            # K (k x fd_dim) is the pseudocount for latent semantic-feature pairs
            K = model.components_
            # X (4339 x k) is the image-semantic distribution (image ID-latent semantic pairs)
            X = model.transform(feature_vectors_shifted)

            all_latent_semantics = {
                "image-semantic": X.tolist(),
                "semantic-feature": K.tolist(),
            }

            dump(
                model,
                f"{sim_type}_sim-{feature_model}-{dim_reduction_method}-{k}-model.joblib",
            )

            # for each latent semantic, sort object-weight pairs by weights in descending order
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
                "semantic-feature": CC.tolist(),
            }

            # for each latent semantic, sort object-weight pairs by weights in ascending order
            displayed_latent_semantics = [
                sorted(
                    list(zip(feature_ids, latent_semantic)),
                    key=lambda x: x[1],
                    reverse=False,
                )[:top_images]
                for latent_semantic in Y.T
            ]
            print("Note: for K-Means we display distances, in ascending order")

    for idx, latent_semantic in enumerate(displayed_latent_semantics):
        print(f"Latent semantic no. {idx}")
        for obj_id, weight in latent_semantic:
            if valid_dim_reduction_methods[dim_reduction_method] == 4:
                print(f"{sim_type}\t{obj_id}\t-\tDistance\t{weight}")
            else:
                print(f"{sim_type}\t{obj_id}\t-\tWeight\t{weight}")

    # Finally also save sim_matrix
    all_latent_semantics["sim-matrix"] = sim_matrix.tolist()

    with open(
        f"{sim_type}_sim-{feature_model}-{dim_reduction_method}-{k}-semantics.json",
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(all_latent_semantics, output_file, ensure_ascii=False)


def find_label_label_similarity(fd_collection, feature_model):
    """
    Calculate similarity between labels. Lower values indicate higher similarities
    """
    assert (
        feature_model in valid_feature_models.values()
    ), "feature_model should be one of " + str(list(valid_feature_models.keys()))

    label_sim_matrix = []
    label_mean_vectors = []

    for label in range(NUM_LABELS):
        # get representative vectors for the label
        label_mean_vectors.append(
            calculate_label_representatives(fd_collection, label, feature_model)
        )

    label_sim_matrix = np.zeros((NUM_LABELS, NUM_LABELS))

    # Calculate half and fill the other
    for i in range(NUM_LABELS):
        for j in range(i + 1, NUM_LABELS):
            # Note: lower the value, lower the distance => higher the similarity
            label_sim_matrix[i][j] = label_sim_matrix[j][i] = feature_distance_matches[
                feature_model
            ](np.array(label_mean_vectors[i]), np.array(label_mean_vectors[j]))
    return label_sim_matrix


def find_image_image_similarity(fd_collection, feature_model):
    """
    Calculate similarity between images. Lower values indicate higher similarities
    """
    assert (
        feature_model in valid_feature_models.values()
    ), "feature_model should be one of " + str(list(valid_feature_models.keys()))

    feature_vectors = [
        np.array(
            img_fds[feature_model]
        ).flatten()  # get the specific feature model's feature vector
        for img_fds in fd_collection.find()  # repeat for all images
    ]
    image_sim_matrix = np.zeros((NUM_IMAGES, NUM_IMAGES))

    # Calculate half and fill the other
    for i in range(NUM_IMAGES):
        for j in range(i + 1, NUM_IMAGES):
            # Note: lower the value, lower the distance => higher the similarity
            image_sim_matrix[i][j] = image_sim_matrix[j][i] = feature_distance_matches[
                feature_model
            ](np.array(feature_vectors[i]), np.array(feature_vectors[j]))
    return image_sim_matrix


def compute_cp_decomposition(fd_collection, feature_model, rank):
    assert (
        feature_model in valid_feature_models.values()
    ), "feature_model should be one of " + str(list(valid_feature_models.keys()))

    all_images = list(fd_collection.find())

    # (images, features, labels)
    data_tensor_shape = (
        NUM_IMAGES,
        np.array(all_images[0][feature_model]).flatten().shape[0],
        NUM_LABELS,
    )
    data_tensor = np.zeros(data_tensor_shape)
    print(data_tensor_shape)

    # create data tensor
    for img_id in range(NUM_IMAGES):
        label = all_images[img_id]["true_label"]
        data_tensor[img_id, :, label] = np.array(
            all_images[img_id][feature_model]
        ).flatten()

    weights_tensor, factor_matrices = tl.decomposition.parafac(
        data_tensor, rank=rank, normalize_factors=True
    )
    return weights_tensor, factor_matrices


def extract_CP_semantics_from_feature_model(
    fd_collection,
    rank,
    feature_model,
    top_images=None,
):
    assert (
        feature_model in valid_feature_models.values()
    ), "feature_model should be one of " + str(list(valid_feature_models.keys()))

    top_img_str = ""
    if top_images is not None:
        top_img_str = f" (showing only top {top_images} image-weight pairs for each latent semantic)"
    print(
        "Applying CP decomposition on the {} space to get {} latent semantics{}...".format(
            feature_model, rank, top_img_str
        )
    )

    all_images = list(fd_collection.find())
    img_ids = [img for img in range(NUM_IMAGES)]
    img_feature_ids = [
        feature_num for feature_num in range(len(all_images[0][feature_model]))
    ]
    img_label_ids = [label for label in range(NUM_LABELS)]
    feature_ids = [img_ids, img_feature_ids, img_label_ids]

    weights_tensor, factor_matrices = compute_cp_decomposition(
        fd_collection, feature_model, rank
    )

    all_latent_semantics = {
        "image-semantic": factor_matrices[0].tolist(),
        "feature-semantic": factor_matrices[1].tolist(),
        "label-semantic": factor_matrices[2].tolist(),
        "semantics-core": weights_tensor.tolist(),
    }

    strs = ["image", "feature", "label"]
    for i in range(3):
        displayed_latent_semantics = [
            sorted(
                list(zip(feature_ids[i], latent_semantic)),
                key=lambda x: x[1],
                reverse=True,
            )[:top_images]
            for latent_semantic in factor_matrices[i].T
        ]
        print(f"Showing {strs[i]}-weight latent semantic")
        for idx, latent_semantic in enumerate(displayed_latent_semantics):
            print(f"Latent semantic no. {idx}")
            for obj_id, weight in latent_semantic:
                print(f"{strs[i]}\t{obj_id}\t-\tweight\t{weight}")

    with open(
        f"{feature_model}-cp-{rank}-semantics.json",
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(all_latent_semantics, output_file, ensure_ascii=False)
