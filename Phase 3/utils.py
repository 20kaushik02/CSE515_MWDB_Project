# All imports
# Math
import math
import random
import cv2
import numpy as np
from scipy.stats import pearsonr

from collections import defaultdict

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

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
    
    def entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))
    
    def information_gain(self, X, y, feature, threshold):
        left_idxs = X[:, feature] <= threshold
        right_idxs = ~left_idxs
        
        left_y = y[left_idxs]
        right_y = y[right_idxs]
        
        p_left = len(left_y) / len(y)
        p_right = len(right_y) / len(y)
        
        gain = self.entropy(y) - (p_left * self.entropy(left_y) + p_right * self.entropy(right_y))
        return gain
    
    def find_best_split(self, X, y):
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self.information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or depth == self.max_depth:
            return Node(value=np.argmax(np.bincount(y)))
        
        best_feature, best_threshold = self.find_best_split(X, y)
        
        if best_feature is None:
            return Node(value=np.argmax(np.bincount(y)))
        
        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = ~left_idxs
        
        left_subtree = self.build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_subtree = self.build_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)
    
    def fit(self, X, y):
        X = np.array(X)  # Convert to NumPy array
        y = np.array(y)  # Convert to NumPy array
        self.tree = self.build_tree(X, y)
    
    def predict_instance(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self.predict_instance(x, node.left)
        else:
            return self.predict_instance(x, node.right)
    
    def predict(self, X):
        X = np.array(X)  # Convert to NumPy array
        predictions = []
        for x in X:
            pred = self.predict_instance(x, self.tree)
            predictions.append(pred)
        return np.array(predictions)

class LSHIndex:
    def __init__(self, num_layers, num_hashes, dimensions, seed=42):
        self.num_layers = num_layers
        self.num_hashes = num_hashes
        self.dimensions = dimensions
        self.index = [defaultdict(list) for _ in range(num_layers)]
        self.hash_functions = self._generate_hash_functions(seed)

    def _generate_hash_functions(self, seed):
        np.random.seed(seed)
        hash_functions = []
        for _ in range(self.num_layers):
            layer_hashes = []
            for _ in range(self.num_hashes):
                random_projection = np.random.randn(self.dimensions)
                random_projection /= np.linalg.norm(random_projection)
                layer_hashes.append(random_projection)
            hash_functions.append(layer_hashes)
        return hash_functions

    def hash_vector(self, vector):
        hashed_values = []
        for i in range(self.num_layers):
            layer_hashes = self.hash_functions[i]
            layer_hash = [int(np.dot(vector, h) > 0) for h in layer_hashes]
            hashed_values.append(tuple(layer_hash))
        return hashed_values

    def add_vector(self, vector, image_id):
        hashed = self.hash_vector(vector)
        for i in range(self.num_layers):
            self.index[i][hashed[i]].append((image_id, vector))

    def query(self, query_vector):
        hashed_query = self.hash_vector(query_vector)
        candidates = set()
        for i in range(self.num_layers):
            candidates.update(self.index[i][hashed_query[i]])
        return candidates

    def query_t_unique(self, query_vector, t):
        hashed_query = self.hash_vector(query_vector)
        candidates = []
        unique_vectors = set()  # Track unique vectors considered

        for i in range(self.num_layers):
            candidates.extend(self.index[i][hashed_query[i]])

        # Calculate Euclidean distance between query and candidate vectors
        distances = []
        for candidate in candidates:
            unique_vectors.add(tuple(candidate[1]))  # Adding vectors to track uniqueness
            # unique_vectors.add((candidate))  # Adding vectors to track uniqueness
            distance = np.linalg.norm(candidate[0] - query_vector)
            distances.append(distance)

        # Sort candidates based on Euclidean distance and get t unique similar vectors
        unique_similar_vectors = []
        for distance, candidate in sorted(zip(distances, candidates)):
            if len(unique_similar_vectors) >= t:
                break
            if tuple(candidate) not in unique_similar_vectors:
                unique_similar_vectors.append(tuple(candidate))

        return list(unique_similar_vectors), len(unique_vectors), len(candidates)
