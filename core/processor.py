# A class to preprocess the malware images to make it usable for training neural networks.
import os

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


class ImageProcessor:

    # Initialization of the image pre-processor with required file locations.
    def __init__(self, train_dir: str, test_dir: str, val_dir: str, size, colormode: str) -> None:
        self.train_gen = None
        self.test_gen = None
        self.val_gen = None
        self.colormode = colormode
        self.training_data_directory = train_dir
        self.testing_data_directory = test_dir
        self.validation_data_directory = val_dir
        self.size = size
        self.batch_size = 32
        self.seed = 42
        self.class_mode = "categorical"

    # Creating generators based on the preprocessing requirements of the CNN architecture.
    def create_generators(self):
        self.train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
        )

        self.test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255
        )

        self.val_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255
        )

    # Reading the images from the respective directories.
    def get_images(self):
        train_images = self.train_gen.flow_from_directory(
            directory=self.training_data_directory,
            target_size=self.size,
            color_mode=self.colormode,
            class_mode=self.class_mode,
            batch_size=self.batch_size,
            shuffle=True,
            seed=self.seed,
            subset='training',
        )

        val_images = self.val_gen.flow_from_directory(
            directory=self.validation_data_directory,
            target_size=self.size,
            classes=sorted([i for i in os.listdir(self.training_data_directory)]),
            color_mode=self.colormode,
            class_mode=self.class_mode,
            batch_size=self.batch_size,
            shuffle=True,
            seed=self.seed
        )

        test_images = self.test_gen.flow_from_directory(
            directory=self.testing_data_directory,
            target_size=self.size,
            color_mode=self.colormode,
            class_mode=self.class_mode,
            batch_size=self.batch_size,
            shuffle=False,
            seed=self.seed
        )

        return train_images, val_images, test_images

    @staticmethod
    def generator_to_numpy(generator):
        """
        Convert a Keras image generator to numpy arrays of data and labels.

        Args:
            generator: Keras ImageDataGenerator flow instance.

        Returns:
            X (np.ndarray): Flattened image data.
            y (np.ndarray): One-hot encoded labels.
        """
        X, y = [], []
        for batch_x, batch_y in generator:
            X.append(batch_x)
            y.append(batch_y)
            if len(X) * generator.batch_size >= generator.n:
                break
        X = np.vstack(X)  # Combine all batches into one array
        y = np.vstack(y)  # Combine all labels into one array

        y = np.argmax(y, axis=1)  # Convert one-hot labels to class indices

        return X, y

    def get_image_flattened(self):
        train_images, val_images, test_images = self.get_images()

        # Convert train, validation, and test data to numpy arrays
        x_train, y_train = ImageProcessor.generator_to_numpy(train_images)
        x_val, y_val = ImageProcessor.generator_to_numpy(val_images)
        x_test, y_test = ImageProcessor.generator_to_numpy(test_images)

        # Flatten the image data for classical models
        x_train = x_train.reshape(x_train.shape[0], -1)  # Flatten each image
        x_val = x_val.reshape(x_val.shape[0], -1)  # Flatten each image
        x_test = x_test.reshape(x_test.shape[0], -1)  # Flatten each image

        # We also scale the data
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)

        return x_train, y_train, x_val, y_val, x_test, y_test
