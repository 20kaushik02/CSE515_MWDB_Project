# Phase 1

Getting started: PyTorch, Caltech101 dataset, ResNet50 and similarity measures

- Refer phase1_project23.pdf for problem description
- For task 3, the best distance measures seem to be:
  - Color moments - Pearson (faces especially)
  - Histogram of oriented gradients (HOG) - Cosine similarity
  - ResNet50 (avgpool, layer3, fc) - unsatisfactory results for all, simply used euclidean

## Requirements and dependencies

- Requires MongoDB server (local or otherwise)
- Install packages from requirements.txt

## Task 1 - task_1.ipynb

After installing, run all cells in the notebook. There will be a prompt to give input for image ID. Range is 0 to 8677

## Task 2 - task_2.ipynb

Dataset processing and storage to database. Ensure MongoDB server is running, modify connection URI as needed if running on Atlas

## Task 3 - task_3.ipynb

Execute all cells till before the "Target images" markdown cell. There will be four prompts to give input for:

- **Image ID**: integer, 0 to 8677
- No. of similar images needed, **k**: positive integer
- **Feature model** - one of ["cm", "hog", "avgpool", "layer3", "fc"]
  - _Note: only hog is applicable for all images. Others cannot be used for grayscale images_
- **Similarity/distance measure** - one of ["euclidean", "cosine", "pearson"]
