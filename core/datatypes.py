import os
import random

from matplotlib import pyplot as plt, image as mpimg
import seaborn as sns


class MalwareImages:
    # A class for loading an image dataset and understanding its class distribution
    # and image samples across classes.

    # Initialize the dataset loader with the dataset path.
    def __init__(self, data_dir: str, n: int):
        self.data_directory = data_dir
        self.class_distribution = dict()
        self.rows = n
        self.columns = n

    # Computation of the class distribution of the dataset.
    def __compute_class_distribution(self):
        for malware_type in os.listdir(self.data_directory):
            malware_img_dir = os.path.join(self.data_directory, malware_type)
            self.class_distribution[malware_type] = len(os.listdir(malware_img_dir))

    # Plotting the class distribution.
    def plot_class_distribution(self):
        self.__compute_class_distribution()

        malware_classes = list(self.class_distribution.keys())
        malware_class_frequency = list(self.class_distribution.values())
        color_palette = sns.color_palette("pastel")
        plt.figure(figsize=(8, 8))
        sns.barplot(y=malware_classes,
                    x=malware_class_frequency,
                    palette=color_palette,
                    edgecolor="black",
                    orient='h')
        plt.title("Malware Class Distribution")
        plt.xlabel("Malware Class Frequency")
        plt.ylabel("Malware Type")

    # Insights into samples of different malware images across different classes.
    def malware_samples(self):
        c = 0
        fig, axs = plt.subplots(self.rows, self.columns, figsize=(15, 15))

        for malware_type in os.listdir(self.data_directory):
            malware_img_dir = os.path.join(self.data_directory, malware_type)
            malware_img_sample = random.choice(list(os.listdir(malware_img_dir)))
            malware_img_sample_path = os.path.join(malware_img_dir, malware_img_sample)
            image = mpimg.imread(malware_img_sample_path)
            axs[c // self.columns, c % self.columns].imshow(image, cmap="gray")
            axs[c // self.columns, c % self.columns].set_title(malware_type)
            c += 1

        fig.suptitle("Sample for Malware types")
        plt.subplots_adjust(wspace=0.9)
        plt.show()
