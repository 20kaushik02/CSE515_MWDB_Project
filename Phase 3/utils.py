# All imports
# Math
import math
import random
import cv2
import numpy as np
from scipy.stats import pearsonr

from collections import defaultdict

from sklearn.decomposition import LatentDirichletAllocation

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
