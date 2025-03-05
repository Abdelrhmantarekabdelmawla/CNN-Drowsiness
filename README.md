# Drowsiness Detection - CNN-Based Model

## Overview
This repository contains a **CNN-based drowsiness detection model** that classifies facial images into two categories: **alert** and **drowsy**. The goal of this project is to enhance driver safety by detecting signs of drowsiness in real time.

## Features
- Deep learning-based image classification for drowsiness detection.
- Trained on a preprocessed dataset with image augmentation.
- Handles variations in lighting and facial angles.
- Can be integrated into real-time applications.

## Dataset
The dataset was generated with the photo_taker file, the result consists of labeled images of individuals in **alert** and **drowsy** states. It has undergone preprocessing steps, including:
- Image resizing
- Normalization
- Data augmentation (flipping, rotation, and contrast adjustment)

## Model Architecture
- **Convolutional Neural Network (CNN)**
- Multiple convolutional and pooling layers
- Fully connected layers for classification

## The download the model weights
https://drive.google.com/file/d/1wFgLetA4iMWipzID_X_QE7XyoEACASL0/view?usp=sharing

## Results
- Achieved **96.1% F1 Score** on the test dataset.
- Performs well under good lighting conditions.
- Sensitive to extreme head positions and occlusions.

## Future Improvements
- Improve robustness to occlusions and poor lighting.
- Enhance real-time performance and reduce inference latency.
- Explore hybrid models combining CNN with facial landmark detection.

## Contributing
Feel free to contribute by submitting issues or pull requests to improve the model and codebase.

## License
This project is licensed under the MIT License.

