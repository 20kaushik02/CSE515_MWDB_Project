# All imports
# Math
import math
import random
import cv2
import numpy as np
from scipy.stats import pearsonr

from collections import defaultdict

from sklearn.decomposition import LatentDirichletAllocation

# from sklearn.cluster import KMeans

# Torch
import torch
import torchvision.transforms as transforms
from torchvision.datasets import Caltech101
from torchvision.models import resnet50, ResNet50_Weights

import tensorly as tl

import heapq

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

NUM_LABELS = 101
NUM_IMAGES = 4338

valid_classification_methods = {
    "m-nn": 1,
    "decision-tree": 2,
    "ppr": 3,
}

def getCollection(db, collection):
    """Load feature descriptor collection from MongoDB"""
    client = MongoClient("mongodb://localhost:27017")
    return client[db][collection]


def euclidean_distance_measure(img_1_fd, img_2_fd):
    img_1_fd_reshaped = img_1_fd.flatten()
    img_2_fd_reshaped = img_2_fd.flatten()

    # Calculate Euclidean distance
    return math.dist(img_1_fd_reshaped, img_2_fd_reshaped)


valid_feature_models = {
    "cm": "cm_fd",
    "hog": "hog_fd",
    "avgpool": "avgpool_fd",
    "layer3": "layer3_fd",
    "fc": "fc_fd",
    "resnet": "resnet_fd",
}



class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = {}

    def calculate_gini(self, labels):
        classes, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        gini = 1 - sum(probabilities ** 2)
        return gini

    def find_best_split(self, data, labels):
        best_gini = float('inf')
        best_index = None
        best_value = None

        for index in range(len(data[0])):
            unique_values = np.unique(data[:, index])
            for value in unique_values:
                left_indices = np.where(data[:, index] <= value)[0]
                right_indices = np.where(data[:, index] > value)[0]

                left_gini = self.calculate_gini(labels[left_indices])
                right_gini = self.calculate_gini(labels[right_indices])

                gini = (len(left_indices) * left_gini + len(right_indices) * right_gini) / len(data)

                if gini < best_gini:
                    best_gini = gini
                    best_index = index
                    best_value = value

        return best_index, best_value

    def build_tree(self, data, labels, depth=0):
        if len(np.unique(labels)) == 1 or (self.max_depth and depth >= self.max_depth):
            return {'class': np.argmax(np.bincount(labels))}

        best_index, best_value = self.find_best_split(data, labels)
        left_indices = np.where(data[:, best_index] <= best_value)[0]
        right_indices = np.where(data[:, best_index] > best_value)[0]

        left_subtree = self.build_tree(data[left_indices], labels[left_indices], depth + 1)
        right_subtree = self.build_tree(data[right_indices], labels[right_indices], depth + 1)

        return {'index': best_index, 'value': best_value,
                'left': left_subtree, 'right': right_subtree}

    def fit(self, data, labels):
        self.tree = self.build_tree(data, labels)

    def predict_sample(self, sample, tree):
        if 'class' in tree:
            return tree['class']
        
        if sample[tree['index']] <= tree['value']:
            return self.predict_sample(sample, tree['left'])
        else:
            return self.predict_sample(sample, tree['right'])

    def predict(self, data):
        predictions = []
        for sample in data:
            prediction = self.predict_sample(sample, self.tree)
            predictions.append(prediction)
        return predictions


class LSH:
    def __init__(self, data, num_layers, num_hashes):
        self.data = data
        self.num_layers = num_layers
        self.num_hashes = num_hashes
        self.hash_tables = [defaultdict(list) for _ in range(num_layers)]
        self.unique_images_considered = set()
        self.overall_images_considered = set()
        self.create_hash_tables()

    def hash_vector(self, vector, seed):
        np.random.seed(seed)
        random_vectors = np.random.randn(self.num_hashes, len(vector))
        return ''.join(['1' if np.dot(random_vectors[i], vector) >= 0 else '0' for i in range(self.num_hashes)])

    def create_hash_tables(self):
        for layer in range(self.num_layers):
            for i, vector in enumerate(self.data):
                hash_code = self.hash_vector(vector, seed=layer)
                self.hash_tables[layer][hash_code].append(i)

    def find_similar(self, external_image, t, threshold=0.9):
        similar_images = set()
        visited_buckets = set()
        unique_images_considered = set()

        for layer in range(self.num_layers):
            hash_code = self.hash_vector(external_image, seed=layer)
            visited_buckets.add(hash_code)

            for key in self.hash_tables[layer]:
                if key != hash_code and self.hamming_distance(key, hash_code) <= 2:
                    visited_buckets.add(key)

                    for idx in self.hash_tables[layer][key]:
                        similar_images.add(idx)
                        unique_images_considered.add(idx)

        self.unique_images_considered = unique_images_considered
        self.overall_images_considered = similar_images

        similarities = [
            (idx, self.euclidean_distance(external_image, self.data[idx])) for idx in similar_images
        ]
        similarities.sort(key=lambda x: x[1])

        return [idx for idx, _ in similarities[:t]]

    def hamming_distance(self, code1, code2):
        return sum(c1 != c2 for c1, c2 in zip(code1, code2))

    def euclidean_distance(self, vector1, vector2):
        return np.linalg.norm(vector1 - vector2)

    def get_unique_images_considered_count(self):
        return len(self.unique_images_considered)

    def get_overall_images_considered_count(self):
        return len(self.overall_images_considered)

def extract_latent_semantics_from_feature_model(
    fd_collection,
    k,
    feature_model,
):
    """
    Extract latent semantics for entire collection at once for a given feature_model and dim_reduction_method, and display the imageID-semantic weight pairs

    Leave `top_images` blank to display all imageID-weight pairs
    """


    label_features = np.array([
        np.array(
            calculate_label_representatives(fd_collection, label, feature_model)
        ).flatten()  # get the specific feature model's feature vector
        for label in range(NUM_LABELS)
          # repeat for all images
    ])

    print(
        "Applying {} on the {} space to get {} latent semantics.".format(
            "svd", feature_model, k
        )
    )

    all_latent_semantics = {}

    
    U, S, V_T = svd(label_features, k=k)

    U = [C.real for C in U]
    S = [C.real for C in S]
    V_T = [C.real for C in V_T]

    all_latent_semantics = {
        "image-semantic": U,
        "semantics-core": S,
        "semantic-feature": V_T,
    }

    # for each latent semantic, sort imageID-weight pairs by weights in descending order
    return all_latent_semantics

    
def calculate_label_representatives(fd_collection, label, feature_model):
    """Calculate representative feature vector of a label as the mean of all feature vectors under a feature model"""

    label_fds = [
        np.array(
            img_fds[feature_model]
        ).flatten()  # get the specific feature model's feature vector
        for img_fds in fd_collection.find(
            {"true_label": label, "$mod": [2,0]}
        )  # repeat for all images
    ]

    # Calculate mean across each dimension
    # and build a mean vector out of these means
    label_mean_vector = [sum(col) / len(col) for col in zip(*label_fds)]
    return label_mean_vector

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