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
import os
from os import getenv
from dotenv import load_dotenv
import warnings
from joblib import dump, load

load_dotenv()

# MongoDB
from pymongo import MongoClient

# Visualizing
import matplotlib.pyplot as plt


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


def euclidean_distance_measure(img_1_fd, img_2_fd):
    img_1_fd_reshaped = img_1_fd.flatten()
    img_2_fd_reshaped = img_2_fd.flatten()

    # Calculate Euclidean distance
    return math.dist(img_1_fd_reshaped, img_2_fd_reshaped)


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

valid_feature_models = {
    "cm": "cm_fd",
    "hog": "hog_fd",
    "avgpool": "avgpool_fd",
    "layer3": "layer3_fd",
    "fc": "fc_fd",
    "resnet": "resnet_fd",
}

def predict_m_nn_classifier(fd_collection, m, feature_model, selected_image_fd):
    """
    Create the m-NN classifier from the selected feature space
    """

    assert (
        feature_model in valid_feature_models.values()
    ), "feature_moel should be one of " + str(list(valid_feature_models.keys()))

    all_images = list(fd_collection.find())
    feature_ids = [img["image_id"] for img in all_images]

    feature_vectors = np.array(
        [np.array(img[feature_model]).flatten() for img in all_images]
    )

    distances = []

    for fd, id in zip(feature_vectors, feature_ids):
        distances.append({"image_id": id, "distance": euclidean_distance_measure(selected_image_fd, fd)})

    distances = sorted(distances, key=lambda x: x["distance"])

    return distances[:10]