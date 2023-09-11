# Phase 1

Getting started: PyTorch, Caltech101 dataset, ResNet50 and similarity measures

- Refer phase1_project23.pdf for problem description
- For task 3, the best distance measures seem to be:
  - Color moments - Pearson (faces especially)
  - Histogram of oriented gradients (HOG) - Cosine similarity
  - ResNet50 (avgpool, layer3, fc) - unsatisfactory results for all, simply used euclidean
