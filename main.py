import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from models import create_model, load_model_with_weights
from utils import create_data_generators, convert_to_tf_dataset
from scripts.train import train_model
from scripts.infer import infer_model

# Paths to the dataset directories
PATH = 'dataset'
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')

# Calculate the total number of images in each directory
total_train = sum([len(files) for r, d, files in os.walk(train_dir)])
total_val = sum([len(files) for r, d, files in os.walk(validation_dir)])
total_test = sum([len(files) for r, d, files in os.walk(test_dir)])

# Model parameters
batch_size = 32
epochs = 20
IMG_HEIGHT = 150
IMG_WIDTH = 150
num_classes = 11 

# Create data generators
train_data_gen, val_data_gen, test_data_gen = create_data_generators(
    train_dir, validation_dir, test_dir, IMG_HEIGHT, IMG_WIDTH, batch_size
)

# Create model
model = create_model(IMG_HEIGHT, IMG_WIDTH, num_classes)

# Check if weights file exists
weights_path = 'model_weights.weights.h5'
if os.path.exists(weights_path):
    # Load model with weights
    model = load_model_with_weights(weights_path, IMG_HEIGHT, IMG_WIDTH, num_classes)

    # Perform inference
    probabilities, predictions = infer_model(weights_path, test_dir, IMG_HEIGHT, IMG_WIDTH, num_classes)
    
    # Mapping class indices to class labels
    labels = list(train_data_gen.class_indices.keys())

    # Function to plot images with predictions
    def plot_images(images_arr, probabilities=None):
        fig, axes = plt.subplots(len(images_arr), 1, figsize=(5, len(images_arr) * 3))
        for img, probability, ax in zip(images_arr, probabilities, axes):
            ax.imshow(img)
            ax.axis('off')
            probability = probability.tolist()
            predicted_label = labels[np.argmax(probability)]
            confidence = max(probability) * 100
            ax.set_title(f"{confidence:.2f}% {predicted_label}")
        plt.show()

    # Test inference on a few images
    test_images, _ = next(test_data_gen)
    plot_images(test_images[:5], probabilities[:5])

    # Calculate model accuracy on test data
    correct_predictions = np.sum(np.argmax(probabilities, axis=1) == test_data_gen.classes)
    accuracy = correct_predictions / total_test
    print(f"Your model correctly identified {accuracy * 100:.2f}% of the images.")
else:
    # Train model
    train_model(model, train_data_gen, val_data_gen, epochs, batch_size, total_val)
    
    # Save model weights
    model.save_weights(weights_path)
   