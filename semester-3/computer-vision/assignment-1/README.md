# Computer Vision: Image Features & Classical Machine Learning

## Project Overview

This project focuses on classical Computer Vision and Machine Learning pipelines for **Satellite Image Classification**, specifically categorizing images into Urban or Natural domains using the EuroSAT dataset. The assignment explores fundamental image processing, handcrafted feature engineering, and robust classification without relying on deep learning architectures.

## Technical Implementation

### Data Preprocessing
- **Resizing**: Standardization of input images to 128x128 pixels.
- **Grayscale Conversion**: Reducing channel complexity for textural analysis.
- **Enhancement**: Histogram Equalization or CLAHE to improve contrast.
- **Noise Reduction**: Gaussian/Median smoothing to remove artifacts.

### Handcrafted Feature Engineering
The core of this project revolves around extracting meaningful intelligence from pixels using classical algorithms.

**Low-Level Features:**
- **Intensity Histograms**: Pixel distribution analysis.
- **Local Binary Patterns (LBP)**: Extracting structural and textural patterns.
- **Edge Detection**: Sobel and Canny edge algorithms for structural outlines.

**Mid-Level Features:**
- **Histogram of Oriented Gradients (HOG)**: Capturing shape and object silhouettes.
- **SIFT/ORB**: Optional advanced descriptors for keypoint matching.

### Dimensionality Reduction & ML Models
- **Feature Selection**: Principal Component Analysis (PCA) and Variance Thresholding to eliminate low-information features.
- **Classification Models**: Training Support Vector Machines (SVM) and Random Forests.
- **Validation**: Strict 5-fold cross-validation to ensure model robustness.

## Evaluation & Metrics

The models are rigorously evaluated using standard classification metrics:
- Accuracy, Precision, Recall, and F1-score.
- **Confusion Matrices** to visualize true vs. predicted categorizations.
- **Real-World Testing**: Evaluation against an unseen external satellite image to test true generalization.

## Learning Outcomes
- **Classical CV Mastery**: Deep understanding of how handcrafted features (HOG, LBP) contribute to scene classification.
- **Pipeline Engineering**: Building an end-to-end ML pipeline from raw pixels to classification outputs.
- **Performance Trade-offs**: Analyzing the strengths and limitations of SVMs and Random Forests on high-dimensional feature spaces.

---

## Real-Life Applications
The concepts of handcrafted feature extraction, edge detection, and classical ML classification applied here are highly practical in scenarios where interpretability counts or computational resources for deep neural networks are limited:

1. **Urban Planning & Development:** Distinguishing between urban build-up and vegetation from aerial imagery to monitor municipal growth.
2. **Agriculture & Forestry:** Segmenting natural forests versus farmlands or tracking deforestation over time using low-cost hardware.
3. **Disaster Management:** Quickly categorizing affected satellite regions (e.g., flooded vs. normal terrain) without requiring heavy GPU clusters to run the inference.
4. **Autonomous Navigation (Robotics):** Real-time lane detection (using edge gradients and Hough transforms) and terrain classification for rovers operating on low-power compute platforms.

---
*Part of M.Tech AI/ML Academic Portfolio - Semester 3 Computer Vision*
