
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG19

def build_vgg19(input_shape=(224, 224, 3), num_classes=3, trainable_layers=4):
    
    base_model = VGG19(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    return base_model
def build_vgg19(input_shape=(224, 224, 3), num_classes=3, trainable_layers=4):
    
    base_model = VGG19(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze most layers
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False

    # Custom head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)

    return model