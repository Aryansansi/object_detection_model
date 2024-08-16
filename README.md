# Image Classification Project

This project implements an image classification pipeline using TensorFlow and Keras, with a pre-trained VGG16 model as the backbone. The project is structured to allow for easy training, inference, and data processing.

## Project Structure

```
├── datasets/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── models/
│   ├── __init__.py
│   └── model.py
│
├── scripts/
│   ├── __init__.py
│   ├── infer.py
│   └── train.py
│
├── utils/
│   ├── __init__.py
│   └── data_processing.py
│
├── main.py
└── README.md
```

### Datasets Folder

- **`datasets/`**: Contains subfolders for training, validation, and test datasets.
  - `train/`: Directory containing training images organized in subfolders by class.
  - `validation/`: Directory containing validation images organized in subfolders by class.
  - `test/`: Directory containing test images organized in subfolders by class.

### Models Folder

- **`models/`**: Contains the model architecture and methods for creating and loading models.
  - `model.py`: Defines the model creation and loading functions using VGG16 as a base.

### Scripts Folder

- **`scripts/`**: Contains scripts for training the model and performing inference.
  - `train.py`: Handles model training with callbacks for learning rate reduction and early stopping.
  - `infer.py`: Performs inference on test images, generates predictions, and visualizes the results.

### Utils Folder

- **`utils/`**: Contains utility functions for data processing.
  - `data_processing.py`: Contains functions for creating data generators for training, validation, and test datasets.

### Main Script

- **`main.py`**: The main script to run the entire pipeline. It checks for pre-existing model weights, trains the model if necessary, performs inference, and visualizes results.

## Setup

### Requirements

- Python 3.8+
- TensorFlow 2.5+
- NumPy
- Matplotlib

### Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

1. **Prepare your datasets**: Ensure your images are organized in the `train`, `validation`, and `test` directories within the `datasets` folder. Each class should have its own subfolder containing the respective images.

2. **Train the model**:
   Run the following command to train the model:
   ```bash
   python main.py
   ```

   This will train the model and save the weights to a file named `model_weights.weights.h5`.

3. **Perform inference**:
   If the weights file exists, the script will load the model and perform inference on the test dataset.

### Customization

- **Model Architecture**: You can modify the architecture in `models/model.py` if you want to experiment with different model configurations.
- **Data Augmentation**: Adjust the data augmentation parameters in `utils/data_processing.py` to fine-tune the training process.

### Results

- The accuracy of the model on the test dataset is printed after inference.
- The model predictions are visualized with confidence scores for a few test images.
