import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from models.model import load_model_with_weights

def infer_model(weights_path, test_dir, img_height, img_width, num_classes):
    """
    Perform inference using the trained model on the test data directory.

    Args:
        weights_path (str): Path to the model weights file.
        test_dir (str): Directory containing test images.
        img_height (int): Height of the input images.
        img_width (int): Width of the input images.
        num_classes (int): Number of classes in the classification task.

    Returns:
        tuple: A tuple containing:
            - probabilities (numpy.ndarray): The predicted probabilities for each class.
            - predictions (numpy.ndarray): The predicted class indices.
    """
    # Create an ImageDataGenerator for test images
    test_image_generator = ImageDataGenerator(rescale=1./255)
    
    test_data_gen = test_image_generator.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )

    # Load the model with weights
    model = load_model_with_weights(weights_path, img_height, img_width, num_classes)

    # Predict on test data
    probabilities = model.predict(test_data_gen)

    # Mapping class indices to class labels
    labels = list(test_data_gen.class_indices.keys())

    # Function to plot images with predictions
    def plot_images(images_arr, probabilities=None):
        num_images = len(images_arr)
        fig, axes = plt.subplots(num_images, 1, figsize=(5, num_images * 3))
        
        if num_images == 1:
            axes = [axes]  # Make it iterable if it's a single Axes object

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
    accuracy = correct_predictions / len(test_data_gen.classes)
    print(f"Your model correctly identified {accuracy * 100:.2f}% of the images.")
    
    return probabilities, np.argmax(probabilities, axis=1)
