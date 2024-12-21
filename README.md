# Convolutional Neural Networks

## Overview
This project involves building a Convolutional Neural Network (CNN) for classifying images of pet breeds using the Oxford-IIIT Pets Dataset. The task involves several steps including data loading, preprocessing, model design, training, and evaluation.

## Task Breakdown

### Task 1 - Load and Prepare the Oxford-IIIT Pets Dataset
1. **Loading the Dataset**  
   The dataset is loaded using `torchvision.datasets.OxfordIIITPet`. The `trainval` partition is used for training and validation, and the `test` partition is used for testing.
   
2. **Count Classes**  
   The dataset contains images of different pet breeds. The number of unique classes (breeds) is determined.

3. **Visualize the Dataset**  
   A random sample image from each class is displayed to get a visual representation of the dataset.

4. **One-Hot Encoding of Labels**  
   The class labels are converted into one-hot encoded format for the classification task.

5. **Resize Images**  
   All images are resized to 128x128 pixels using `torchvision.transforms.Resize` with bicubic interpolation.

6. **Class Distribution**  
   A bar graph is plotted to show the distribution of classes and to analyze if the dataset is balanced or imbalanced.

### Task 2 - Data Splitting
1. **Split Dataset**  
   The `trainval` partition is divided into training (80%) and validation (20%) sets. The `test` partition is used for evaluating the final model. The split ensures balanced class distributions.

### Task 3 - Design a CNN Model
1. **Model Architecture**  
   A Convolutional Neural Network (CNN) is designed with the following layers:
   - Convolutional layers with appropriate kernel sizes and filters
   - MaxPooling layers
   - Fully connected layers for classification

### Task 4 - Training the Model
1. **Loss Function**  
   The loss function used for this classification task is Cross-Entropy Loss.

2. **Training Setup**  
   - Optimizer: Stochastic Gradient Descent (SGD) with an appropriate learning rate and momentum.
   - Learning Rate Scheduling: `torch.optim.lr_scheduler.ReduceLROnPlateau` to adjust the learning rate based on validation loss.
   - Batch Size: A suitable batch size is chosen for training.
   - During training, both the training and validation losses are tracked.

### Task 5 - Evaluate the Model
1. **Test Set Performance**  
   The trained model is evaluated on the test set. The performance is measured using accuracy and a confusion matrix, which is generated and displayed to analyze the model's predictions.

## Requirements
- Python 3.x
- PyTorch
- torchvision
- Matplotlib
- NumPy

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cnn-oxford-pets.git
