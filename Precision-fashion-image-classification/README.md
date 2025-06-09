# Precision Fashion Image Classification

A computer vision system for fashion product classification using HOG features and KNN classifier.

## Overview

This project implements an image classification system for fashion products using Histogram of Oriented Gradients (HOG) features and K-Nearest Neighbors (KNN) classifier. It achieves high accuracy in classifying different types of fashion products.

## Features

- High accuracy (98%) in fashion product classification
- Fast inference using HOG features
- Simple and interpretable model
- No deep learning required
- Works well with limited computational resources

## Dataset

The project uses a fashion product dataset containing:
- Multiple fashion categories
- Product images
- Balanced class distribution
- High-quality images

## Model Architecture

- **Feature Extraction**: HOG (Histogram of Oriented Gradients)
  - Cell size: 8x8
  - Block size: 2x2
  - Number of bins: 9

- **Classifier**: K-Nearest Neighbors (KNN)
  - Number of neighbors: 5
  - Distance metric: Euclidean
  - Weight: Uniform

## Performance

The model achieves:
- 98% classification accuracy
- Fast inference time
- Good generalization
- Robust to variations in images

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook image-classification-using-hog-knn-98-acc.ipynb
```

2. Run the cells in sequence to:
   - Load and preprocess data
   - Extract HOG features
   - Train the KNN classifier
   - Evaluate the model
   - Make predictions

## Requirements

- Python 3.8+
- OpenCV
- scikit-learn
- NumPy
- Jupyter Notebook

## Directory Structure

```
Precision-fashion-image-classification/
├── fashion_product_dataset_unzip/  # Dataset directory
├── image-classification-using-hog-knn-98-acc.ipynb  # Main notebook
└── Precision-Fashion-HOG-and-KNN-for-Image-Classification.pptx  # Presentation
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. 