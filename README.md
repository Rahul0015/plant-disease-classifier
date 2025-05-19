# Plant Disease Classifier Using CNN

## Problem Statement

A research team aims to develop a Convolutional Neural Network (CNN) model for the automated diagnosis of multiple plant diseases from high-resolution leaf images captured under varying environmental conditions. The dataset comprises images of healthy leaves and leaves exhibiting symptoms of five distinct fungal diseases.

## Solution Overview

This project implements a CNN-based approach to classify plant leaf images into six categories: five fungal diseases and healthy leaves. The model leverages transfer learning with a pre-trained ResNet50 architecture, fine-tuned to adapt to the specific classification task.

## Architecture Design and Justification

- **Pre-trained Model**: Utilized ResNet50, known for its deep architecture and residual connections, which help in training deeper networks effectively.
- **Custom Classifier**:
  - **Fully Connected Layers**: Added layers with ReLU activation to learn complex patterns specific to plant diseases.
  - **Batch Normalization**: Applied to stabilize and accelerate training.
  - **Dropout Layers**: Incorporated to prevent overfitting by randomly deactivating neurons during training.
- **Output Layer**: Configured with six neurons corresponding to the six classes, using softmax activation for multi-class classification.

## Handling Environmental Variability

To enhance the model's robustness against variations in illumination, angle, and background, the following data augmentation techniques were employed:

- **Random Resized Crop**: Simulates zooming in and out.
- **Random Horizontal Flip**: Mimics flipping of leaves.
- **Random Rotation**: Accounts for different leaf orientations.
- **Color Jitter**: Adjusts brightness, contrast, saturation, and hue to simulate varying lighting conditions.

These augmentations help the model generalize better by exposing it to a diverse set of scenarios during training.

## Installation and Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/plant-disease-classifier.git
   cd plant-disease-classifier
   ```
   <<<<<<< HEAD
   Your local README content
   =======
   Remote README content (on GitHub)
   > > > > > > > origin/main
