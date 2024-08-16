import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.applications import VGG16 # type: ignore

def create_model(img_height, img_width, num_classes):
    base_model = VGG16(input_shape=(img_height, img_width, 3),
                       include_top=False,
                       weights='imagenet')
    base_model.trainable = False  # Freeze the base model

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')  # Multi-class classification
    ])

    model.build((None, img_height, img_width, 3))  # Explicitly define the input shape
    return model

def load_model_with_weights(weights_path, img_height, img_width, num_classes):
    base_model = VGG16(input_shape=(img_height, img_width, 3),
                       include_top=False,
                       weights='imagenet')
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.build((None, img_height, img_width, 3))  # Explicitly define the input shape
    model.load_weights(weights_path)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    
    return model
