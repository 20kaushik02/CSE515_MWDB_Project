import json
import tensorly as tl
import numpy as np
from pymongo import MongoClient
from phase1_mongodb import *

client = MongoClient()
client = MongoClient(host="localhost", port=27017)

# Select the database
db = client.Multimedia_Web_DBs

caltechDataset = loadDataset()
num_labels = 101
# Fetch all documents from the collection and then sort them by "_id"
feature_descriptors = list(db.Feature_Descriptors.find({}))
feature_descriptors = sorted(list(db.Feature_Descriptors.find({})), key=lambda x: x["_id"], reverse=False)
label_ids = [x["label"] for x in feature_descriptors]

def compute_cp_decomposition(feature_model, rank):
    
    label_vectors = [(x["label"], x[feature_model]) for x in feature_descriptors if x["_id"] % 2 == 0]

    num_labels = 101
    tensor_shape = (len(label_vectors), len(feature_descriptors[0][feature_model]), num_labels)
    tensor = np.zeros(tensor_shape)
    for id in range(len(label_vectors)):
        label = label_vectors[id][0]
        tensor[id, :, label] = label_vectors[id][1]
    
    weights, factors = tl.decomposition.parafac(tensor, rank=rank, normalize_factors=True)
    return weights, factors


def main():

    
    
    # Step 4: Perform CP-decomposition (parafac) to extract latent semantics
   
    features = ['color_moments', 'hog', 'layer3', 'avgpool', 'fc']

    # User input for feature model to extract
    print("1: Color moments")
    print("2: HOG")
    print("3: Resnet50 Avgpool layer")
    print("4: Resnet50 Layer 3")
    print("5: Resnet50 FC layer")
    feature_model = features[int(input("Select the feature model: ")) - 1]
    k = int(input("Enter k: "))
    weights, factors = compute_cp_decomposition(feature_model, k)
    k_latent_semantics = list(zip(label_ids, factors[0].tolist()))
    k_latent_semantics_display = sorted(list(zip(label_ids, factors[0].tolist())), key = lambda x: x[1][0], reverse = True)
    k_latent_semantics_display = [{"_id": item[0], "semantics": item[1]} for item in k_latent_semantics_display]
    filename = f'{feature_model}-CP-semantics-{k}.json'
    k_latent_semantics = [{"_id": item[0], "semantics": item[1]} for item in k_latent_semantics]
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(k_latent_semantics, f, ensure_ascii = False)
    
    print(k_latent_semantics_display)


if __name__ == "__main__":
   main()