# Image_CLASSIFIER
CIFAR-10 Image Classifier 

This project is a Convolutional Neural Network (CNN) built to classify images from the CIFAR-10 dataset into 10 distinct categories: Plane, Car, Bird, Cat, Deer, Dog, Frog, Horse, Ship, and Truck. The model is implemented using Python and TensorFlow/Keras and includes a feature to predict classes of custom images.

Features

Dataset: CIFAR-10, containing 60,000 32x32 color images in 10 classes.

Model Architecture:

Three convolutional layers with ReLU activation.

Max-pooling layers to reduce spatial dimensions.

Dense layers with a softmax output for classification.

Custom Image Prediction: Load and classify external images (e.g., horse.jpg).

Visualization: Plots the first 16 training images with their respective class names.

Project Structure

cifar10_image_classifier.py: Main Python script containing the model implementation and training process.

horse.jpg: Example custom image to test predictions.

image_classifier.model: Saved model file after training.

Setup Instructions

Prerequisites

Ensure you have the following installed:

Python 3.8 or later

TensorFlow 2.x

OpenCV

NumPy

Matplotlib

Installation

Clone this repository:

git clone 

Install the required dependencies:

pip install -r requirements.txt

Usage

Training the Model
Run the script to train the model:

python cifar10_image_classifier.py

This will train the CNN on the CIFAR-10 dataset and save the trained model as image_classifier.model.

Testing with Custom Images
Replace horse.jpg with your own image file, ensuring it’s resized to 32x32 pixels. The model will predict the class of the image.

Model Performance

The model is trained on a reduced dataset (20,000 images for training and 4,000 for testing) for faster testing. Evaluate its performance using the CIFAR-10 test set:

Loss: Accuracy:

Example Prediction

Prediction: Horse

To Do

Train the model on the full CIFAR-10 dataset for better accuracy.

Enhance the model by adding data augmentation.

Optimize hyperparameters for improved performance.

License

This project is licensed under the MIT License. See LICENSE for more details.

Acknowledgments

CIFAR-10 Dataset

TensorFlow/Keras Documentation
