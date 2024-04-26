
import tensorflow as tf
from tensorflow.keras import layers
import glob


class EndToEndModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # Convolutional layers
        self.conv_layers = tf.keras.Sequential([
            # First convolutional layer, Width: 200, Height: 66, Channels: 3
            layers.InputLayer((66, 200, 3)),
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


def parse_tfrecord_fn(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'steering': tf.io.FixedLenFeature([], tf.float32),
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [66, 200, 3])
    image = tf.image.rgb_to_yuv(image)  # Convert RGB image to YUV
    image = image / 255.0  # Normalize image to [0, 1]
    return image, example['steering']


def load_dataset(tfrecord_files):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn)
    return parsed_dataset


def main():
    tfrecord_files = list(glob.glob("/tf/datasets/*.tfrecord"))
    parsed_dataset = load_dataset(tfrecord_files)
    dataset = (
        parsed_dataset
        .shuffle(buffer_size=1024)
        .prefetch(tf.data.AUTOTUNE)
        .batch(32, drop_remainder=True)
    )

    # Initialize your model
    model = EndToEndModel()

    # Compile the model (make sure to specify the loss and optimizer)
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Step 5: Train the Model
    model.fit(dataset, epochs=10)


if __name__ == "__main__":
    main()
