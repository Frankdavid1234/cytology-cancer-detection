
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_control_cnn(input_shape=(224, 224, 3), num_classes=3):
    
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape,
                      kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Flatten(),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model