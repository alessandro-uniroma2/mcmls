import json
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, \
    confusion_matrix
from typing import List, Dict

from core.dataloader import DatasetLoader
from core.processor import ImageProcessor
from factory.models import ModelFactory
import seaborn as sns


class McMls:
    def __init__(self, results_dir: str):
        """
        Initialize the pipeline.

        Args:
            results_dir: Path to store results.
        """
        self.loader: DatasetLoader = DatasetLoader()
        self.results_directory = Path(self.loader.config.get("GLOBAL", "results"))
        self.results_directory.mkdir(exist_ok=True)
        self.test_data = {dataset: {} for dataset in self.loader.list_datasets()}
        self.mc = None
        self.dmc = None
        self.__supported_models: List[str] = ModelFactory.list_all()

        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.results = {dataset: {} for dataset in self.loader.list_datasets()}

    def list_models(self) -> List[str]:
        """Return a list of supported models."""
        return list(self.__supported_models)

    def preprocess_dataset(self, dataset_name: str):
        """
        Preprocess a dataset using the ImageProcessor class.

        Args:
            dataset_name: Name of the dataset to preprocess.

        Returns:
            Tuple of flattened train, validation, and test data.
        """
        image_flat_size = self.loader.get_dataset_image_size(dataset_name)
        color_mode = self.loader.get_dataset_image_colormode(dataset_name)

        # Preprocessing the images.
        preprocessor = ImageProcessor(
            str(self.loader.get_training_data_dir(dataset_name, split=True)),
            str(self.loader.get_testing_data_dir(dataset_name)),
            str(self.loader.get_validation_data_dir(dataset_name)),
            image_flat_size,
            color_mode
        )
        preprocessor.create_generators()

        # Get flattened data
        return preprocessor.get_image_flattened()

    def train_and_evaluate(self, model_name: str, dataset_name: str):
        """
        Train and evaluate a model on the given dataset.

        Args:
            model_name: Name of the model to train.
            dataset_name: Name of the dataset to use.
        """
        # Preprocess dataset
        x_train, y_train, x_val, y_val, x_test, y_test = self.preprocess_dataset(dataset_name)

        # Initialize and train the model
        model = ModelFactory.from_name(model_name)
        model.create_model()

        # We try to balance it
        x_train, y_train = model.smote(x_train, y_train)

        model.fit(x_train, y_train)

        val_predictions = model.predict(x_val)
        test_predictions = model.predict(x_test)

        # Compute validation metrics
        val_accuracy = accuracy_score(y_val, val_predictions)
        val_precision = precision_score(y_val, val_predictions, average="weighted", zero_division=1)
        val_recall = recall_score(y_val, val_predictions, average="weighted", zero_division=1)
        val_f1 = f1_score(y_val, val_predictions, average="weighted", zero_division=1)

        # Compute test metrics
        test_accuracy = accuracy_score(y_test, test_predictions)
        test_precision = precision_score(y_test, test_predictions, average="weighted", zero_division=1)
        test_recall = recall_score(y_test, test_predictions, average="weighted", zero_division=1)
        test_f1 = f1_score(y_test, test_predictions, average="weighted", zero_division=1)

        # Classification report
        report = classification_report(y_test, test_predictions, output_dict=True, zero_division=1)

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, test_predictions)

        # Save results
        self.results[dataset_name][model_name] = {
            "validation_metrics": {
                "accuracy": val_accuracy,
                "precision": val_precision,
                "recall": val_recall,
                "f1_score": val_f1,
            },
            "test_metrics": {
                "accuracy": test_accuracy,
                "precision": test_precision,
                "recall": test_recall,
                "f1_score": test_f1,
            },
            "classification_report": report,
            "confusion_matrix": conf_matrix.tolist(),  # Convert to list for JSON serialization
        }

    def evaluate_all_models(self):
        """Evaluate all models on all datasets."""
        for dataset in self.loader.list_datasets():
            for model_name in self.list_models():
                try:
                    print(f"Training {model_name} on {dataset}...")
                    self.train_and_evaluate(model_name, dataset)
                except Exception as e:
                    print(f"Exception occurred training {model_name} on {dataset}: {e}")

    def save_results(self):
        """Save all evaluation results as JSON."""
        for dataset, dataset_results in self.results.items():
            dataset_dir = self.results_dir.joinpath(dataset)
            dataset_dir.mkdir(parents=True, exist_ok=True)

            for model_name, model_results in dataset_results.items():
                model_path = dataset_dir.joinpath(f"{model_name}_results.json")
                with open(model_path, "w") as f:
                    json.dump(model_results, f)

    def load_results(self):
        """Load previously saved results into memory."""
        for dataset in self.loader.list_datasets():
            dataset_dir = self.results_dir.joinpath(dataset)
            if not dataset_dir.is_dir():
                continue

            for model_file in dataset_dir.glob("*_results.json"):
                model_name = model_file.stem.replace("_results", "")
                with open(model_file, "r") as f:
                    self.results[dataset][model_name] = json.load(f)

    def summarize_results(self):
        """
        Summarize and visualize the results collected in self.results.

        Produces:
        - A summary table of key metrics (accuracy, precision, recall, F1-score).
        - A bar plot comparing test accuracy across models and datasets.
        - A heatmap for F1-scores across models and datasets.
        """
        # Collect data into a DataFrame for easy visualization
        summary_data = []
        for dataset_name, models in self.results.items():
            for model_name, metrics in models.items():
                test_metrics = metrics["test_metrics"]
                summary_data.append({
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Accuracy": test_metrics["accuracy"],
                    "Precision": test_metrics["precision"],
                    "Recall": test_metrics["recall"],
                    "F1-Score": test_metrics["f1_score"]
                })

        # Create a DataFrame for easier manipulation
        df = pd.DataFrame(summary_data)

        # Display the summary table
        print("Summary of Results:")
        print(df)

        # Bar plot for accuracy comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Dataset", y="Accuracy", hue="Model", errorbar=None)
        plt.title("Test Accuracy Comparison")
        plt.ylabel("Accuracy")
        plt.xlabel("Dataset")
        plt.legend(loc="lower right")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Heatmap for F1-Scores
        f1_pivot = df.pivot(columns=["Model", "Dataset", "F1-Score"])
        plt.figure(figsize=(8, 6))
        sns.heatmap(f1_pivot, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("F1-Score Heatmap")
        plt.ylabel("Model")
        plt.xlabel("Dataset")
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix_from_results(self, dataset_name: str, model_name: str, class_names: List[str]):
        """
        Plot the confusion matrix for a specific dataset and model using self.results.

        Args:
            dataset_name (str): Name of the dataset.
            model_name (str): Name of the model.
            class_names (List[str]): List of class names.
        """
        if dataset_name not in self.results:
            print(f"Dataset '{dataset_name}' not found in results.")
            return
        if model_name not in self.results[dataset_name]:
            print(f"Model '{model_name}' not found for dataset '{dataset_name}'.")
            return

        # Retrieve confusion matrix
        conf_matrix = self.results[dataset_name][model_name]["confusion_matrix"]

        # Plot using seaborn
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title(f"Confusion Matrix: {model_name} on {dataset_name}")
        plt.tight_layout()
        plt.show()

    def compare_models(self, dataset: str):
        """
        Compare performance of all models for a given dataset.

        Args:
            dataset: Name of the dataset.
        """
        if dataset not in self.results:
            print(f"No results for dataset: {dataset}")
            return

        print(f"Comparison for {dataset}:\n")
        for model_name, model_results in self.results[dataset].items():
            print(f"Model: {model_name}")
            print(f"Validation Accuracy: {model_results['validation_metrics']['accuracy']:.2f}")
            print(f"Test Accuracy: {model_results['test_metrics']['accuracy']:.2f}\n")

    def compare_datasets(self, model_name: str):
        """
        Compare performance of a specific model across datasets.

        Args:
            model_name: Name of the model.
        """
        if model_name not in self.list_models():
            print(f"No such model: {model_name}")
            return

        print(f"Comparison for model: {model_name}\n")
        for dataset, dataset_results in self.results.items():
            if model_name not in dataset_results:
                print(f"Model not evaluated on dataset: {dataset}")
                continue

            print(f"Dataset: {dataset}")
            print(f"Validation Accuracy: {dataset_results[model_name]['validation_metrics']['accuracy']:.2f}")
            print(f"Test Accuracy: {dataset_results[model_name]['test_metrics']['accuracy']:.2f}\n")


if __name__ == "__main__":
    # Initialize the pipeline with the DataLoader and ImageProcessor
    loader = DatasetLoader()
    pipeline = McMls(results_dir="results")

    # Train and evaluate all models
    pipeline.evaluate_all_models()

    # Save results to disk
    pipeline.save_results()

    # Compare models for a specific dataset
    for d in loader.list_datasets():
        pipeline.compare_models(d)

    # Compare datasets for a specific model
    for model in pipeline.list_models():
        pipeline.compare_datasets(model)
