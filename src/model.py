import tensorflow as tf
from tensorflow.keras import layers


class EndToEndModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # Convolutional layers
        self.conv_layers = tf.keras.Sequential([
            # First convolutional layer, Width: 200, Height: 66, Channels: 3
            layers.InputLayer(shape=(66, 200, 3)),
            layers.Conv2D(24, kernel_size=5, strides=2, padding='valid'),
            layers.BatchNormalization(),
            layers.ReLU(),

            # Second convolutional layer
            layers.Conv2D(36, kernel_size=5, strides=2, padding='valid'),
            layers.BatchNormalization(),
            layers.ReLU(),

            # Third convolutional layer
            layers.Conv2D(48, kernel_size=5, strides=2, padding='valid'),
            layers.BatchNormalization(),
            layers.ReLU(),

            # Fourth convolutional layer
            layers.Conv2D(64, kernel_size=3, strides=1, padding='valid'),
            layers.BatchNormalization(),
            layers.ReLU(),

            # Fifth convolutional layer
            layers.Conv2D(64, kernel_size=3, strides=1, padding='valid'),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])

        # Flat and fully connected layers
        self.flat_layers = tf.keras.Sequential([
            # Flatten
            layers.Flatten(),
            layers.Dropout(0.5),

            # First fully connected layer
            layers.Dense(1164),
            layers.BatchNormalization(),
            layers.ReLU(),

            # Second fully connected layer
            layers.Dense(100),
            layers.BatchNormalization(),
            layers.ReLU(),

            # Third fully connected layer
            layers.Dense(50),
            layers.BatchNormalization(),
            layers.ReLU(),

            # Fourth fully connected layer
            layers.Dense(10),
            layers.ReLU(),  # Assuming a ReLU here for consistency with the PyTorch model

            # Output layer
            layers.Dense(1)
        ])

    def call(self, inputs):
        x = self.conv_layers(inputs)
        y = self.flat_layers(x)
        return y
