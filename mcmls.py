import json
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import OptimizeWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, \
    confusion_matrix
from typing import List, Dict, Tuple, Any, Optional

from core.dataloader import DatasetLoader
from core.features import FeatureProcessor, OpOption
from core.processor import ImageProcessor
from factory.models import ModelFactory

# Will apply warning filters inside worker processes
os.environ["PYTHONWARNINGS"] = (
    "ignore::UserWarning"
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)


class McMls:
    def __init__(self, results_dir: str, optimize_metric="balanced_accuracy"):
        """
        Initialize the pipeline with enhanced evaluation capabilities.

        Args:
            results_dir: Path to store results.
        """
        print("[*] Patching scikit-learn...")
        from sklearnex import patch_sklearn
        patch_sklearn()
        print("[+] Patched scikit-learn")

        self.loader: DatasetLoader = DatasetLoader()
        self.results_directory = Path(self.loader.config.get("GLOBAL", "results"))
        self.results_directory.mkdir(exist_ok=True)

        self.test_data = {dataset: {} for dataset in self.loader.list_datasets()}
        self.mc = None
        self.dmc = None
        self.__supported_models: List[str] = ModelFactory.list_all()

        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize results dictionary with a more comprehensive structure
        self.results = {dataset: {} for dataset in self.loader.list_datasets()}

        # Load any existing results
        self.load_results()

        # Set metric to optimize
        self.optimize_metric = optimize_metric

    def list_models(self) -> List[str]:
        """Return a list of supported models."""
        return list(self.__supported_models)

    def preprocess_dataset(self, dataset_name: str, option: OpOption = None) -> (
            Tuple
    )[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess a dataset using the ImageProcessor class.

        Args:
            :param dataset_name: Name of the dataset to preprocess.
            :param option: Option to preprocess the dataset

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

        if option.feature_extraction != "raw":
            return preprocessor.get_image_numpy()
        return preprocessor.get_image_flattened()

    def train_and_evaluate(self, model_name: str, dataset_name: str, option: OpOption = None):
        """
        Train and evaluate a model on the given dataset using enhanced evaluation metrics.

        Args:
            model_name: Name of the model to train.
            dataset_name: Name of the dataset to use.
            option: Optional OpOption for feature-based strategy.
        """
        # Get the number of classes for this dataset
        n_classes = len(self.loader.get_dataset_class_names(dataset_name))

        # Create dataset-specific output directory
        dataset_output_dir = self.results_dir.joinpath(dataset_name,
                                                       model_name,
                                                       option.dim_reduction,
                                                       option.feature_extraction)
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize the model
        _model = ModelFactory.from_name(model_name)  # number_of_classes=n_classes
        if _model is None:
            print("[-] Error initializing the model")
            exit(1)

        # Inject preprocessing strategy if provided

        if option:
            _model.option = option
            _model.preprocessor = FeatureProcessor(option)

        # Preprocess Dataset
        print(f"(*) Preprocessing datasets using {option}")
        x_train, y_train, x_val, y_val, x_test, y_test = self.preprocess_dataset(dataset_name, option=option)

        # Create the model
        _model.create_model()

        # Cross-validation evaluation
        print(f"\nPerforming cross-validation evaluation for {model_name} on {dataset_name}...")
        metrics = ['balanced_accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
        cv_results = _model.evaluate(x_train, y_train, dataset_name=dataset_name, metrics=metrics)

        # Grid search
        best_params = None
        param_grid = self._get_param_grid_for_model(model_name)
        if param_grid:
            print(f"\nPerforming grid search for {model_name} on {dataset_name}...")
            best_params, tuned_model = _model.grid_search(
                x_train,
                y_train,
                param_grid=param_grid,
                dataset_name=dataset_name,
                plot=True,
                output_dir=str(dataset_output_dir),
                scoring=self.optimize_metric
            )
            _model.model = tuned_model

        # Learning curve
        print(f"\nGenerating learning curve for {model_name} on {dataset_name}...")
        _model.plot_learning_curve(
            x_train,
            y_train,
            dataset_name=dataset_name,
            output_dir=str(dataset_output_dir)
        )

        # Train
        print(f"\nTraining {model_name} on {dataset_name}...")
        _model.fit(x_train, y_train)

        # Validation predictions
        print(f"\nEvaluating {model_name} on {dataset_name} validation set...")
        val_predictions = _model.predict(x_val)

        # Test predictions and set metrics
        test_results = _model.evaluate_test_set(
            x_test,
            y_test,
            dataset_name=dataset_name,
            output_dir=str(dataset_output_dir)
        )

        # Validation metrics
        val_accuracy = accuracy_score(y_val, val_predictions)
        val_precision = precision_score(y_val, val_predictions, average="weighted", zero_division=1)
        val_recall = recall_score(y_val, val_predictions, average="weighted", zero_division=1)
        val_f1 = f1_score(y_val, val_predictions, average="weighted", zero_division=1)
        val_balanced_accuracy = recall_score(y_val, val_predictions, average="macro", zero_division=1)

        # Store results
        self.results[dataset_name][model_name] = {}
        self.results[dataset_name][model_name][option.dim_reduction] = {}
        self.results[dataset_name][model_name][option.dim_reduction][option.feature_extraction] = {
            "validation_metrics": {
                "accuracy": val_accuracy,
                "balanced_accuracy": val_balanced_accuracy,
                "precision": val_precision,
                "recall": val_recall,
                "f1_score": val_f1,
            },
            "best_params": best_params,
            "cross_validation": cv_results,
            "test_metrics": {
                "accuracy": test_results["accuracy"],
                "precision": test_results["classification_report"]["weighted avg"]["precision"],
                "recall": test_results["classification_report"]["weighted avg"]["recall"],
                "f1_score": test_results["classification_report"]["weighted avg"]["f1-score"],
                "balanced_accuracy": test_results["classification_report"]["macro avg"]["recall"],
            },
            "classification_report": test_results["classification_report"],
            "confusion_matrix": test_results["confusion_matrix"].tolist(),
            "normalized_confusion_matrix": test_results["normalized_confusion_matrix"].tolist(),
        }

        def sanitize(obj):
            """Recursively convert NumPy types and arrays to native Python types."""
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize(i) for i in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64, np.int64, np.int32)):
                return obj.item()
            else:
                return obj

        # Save model evaluation
        _model.save_evaluation_results(dataset_output_dir.joinpath("model_evaluation_results.json"))
        self.save_results()

        return _model

    def evaluate_all_models(self, early_save=True, force_retrain=False, option: OpOption = None):
        """Evaluate all models on all datasets."""
        for model_name in self.list_models():
            self.evaluate_model(model_name, early_save=early_save, force_retrain=force_retrain, option=option)

    def evaluate_model(self, model_name, early_save=True, force_retrain=False, option: OpOption = None):
        """Evaluate a specific model on all datasets."""
        for dataset in self.loader.list_datasets():
            try:
                print(f"\n=== Training {model_name} on {dataset} - ({option}) ===\n")

                if not force_retrain and self.check_results(dataset, model_name, option.dim_reduction,
                                                            option.feature_extraction):
                    print(f"Skipping evaluation of {model_name} on {dataset} - ({option}) (results already exist)")
                    continue

                self.train_and_evaluate(model_name, dataset, option=option)

                if early_save:
                    self.save_results()

            except Exception as e:
                print(f"Exception occurred training {model_name} on {dataset} - ({option}): {e}")
                import traceback
                traceback.print_exc()

    def evaluate_subset(self, model_names: list, dataset_names: list, early_save=True, force_retrain=False,
                        option: OpOption = None):
        """Evaluate a subset of models on specific datasets."""
        for model_name in model_names:
            for dataset_name in dataset_names:
                try:
                    print(f"\n=== Training {model_name} on {dataset_name} - ({option}) ===\n")
                    if not force_retrain and self.check_results(dataset_name, model_name, option.dim_reduction,
                                                                option.feature_extraction):
                        print(
                            f"Skipping evaluation of {model_name} on {dataset_name} - ({option}) (results already exist)")
                        continue

                    self.train_and_evaluate(model_name, dataset_name, option=option)

                    if early_save:
                        self.save_results()

                except Exception as e:
                    print(f"Exception occurred training {model_name} on {dataset_name} - ({option}): {e}")
                    import traceback
                    traceback.print_exc()

    def check_results(self, dataset_name: str, model_name: str, dim_reduction: str, feature_extraction: str) -> bool:
        return (
                feature_extraction in self.results.get(dataset_name, {})
                .get(model_name, {})
                .get(dim_reduction, {})
        )

    def save_results(self):
        """Save all evaluation results as JSON, with strategy-aware filenames."""

        def sanitize_for_json(obj, strict=True):
            """
            Recursively convert a Python object containing NumPy types to be JSON-serializable.

            Parameters:
                obj (Any): The input object (dict, list, scalar, etc.)
                strict (bool): If True, raise error on unserializable values.
                               If False, convert unrecognized objects to strings.

            Returns:
                A sanitized version of the object, safe for json.dumps().
            """
            if isinstance(obj, dict):
                return {k: sanitize_for_json(v, strict=strict) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_for_json(item, strict=strict) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(sanitize_for_json(item, strict=strict) for item in obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            else:
                if strict:
                    raise TypeError(f"Object of type {type(obj)} is not JSON serializable: {obj}")
                else:
                    return str(obj)

        results = sanitize_for_json(self.results)

        for dataset, model_dict in results.items():
            dataset_dir = self.results_dir.joinpath(dataset)
            dataset_dir.mkdir(parents=True, exist_ok=True)

            for model_name, strategy_dict in model_dict.items():
                dataset_model_dir = dataset_dir.joinpath(model_name)
                for dim_reduction, mech_dict in strategy_dict.items():
                    dim_reduction_dir = dataset_model_dir.joinpath(dim_reduction)
                    for feature_extraction, model_results in mech_dict.items():
                        feature_extraction_dir = dim_reduction_dir.joinpath(feature_extraction)
                        model_tag = f"{model_name}_{dim_reduction.lower()}_{feature_extraction.lower()}"
                        model_path = feature_extraction_dir.joinpath(f"{model_tag}_results.json")
                        try:
                            with open(model_path, "w") as f:
                                json.dump(model_results, f, indent=2)
                        except Exception as e:
                            print(f"Exception occurred saving results to {model_path}: {e}")

    def load_results(self):
        for dataset in self.loader.list_datasets():
            dataset_dir = self.results_dir.joinpath(dataset)
            if not dataset_dir.is_dir():
                continue

            for model_file in dataset_dir.rglob("*_results.json"):
                if model_file.name == "model_evaluation_results.json":
                    continue
                filename = model_file.stem.replace("_results", "")
                mechanism = model_file.parent.name
                strategy = model_file.parent.parent.name
                model_name = model_file.parent.parent.parent.name

                print(f"[*] Loading results for {model_name} trained on {dataset}, preprocessed with {strategy} and {mechanism}")

                if model_name not in self.list_models():
                    continue

                with open(model_file, "r") as f:
                    try:
                        metrics = json.load(f)
                        self.results.setdefault(dataset, {}) \
                            .setdefault(model_name, {}) \
                            .setdefault(strategy, {})[mechanism] = metrics
                    except Exception as e:
                        print(f"Could not load {model_file}: {e}")

    def summarize_results(self, metric="balanced_accuracy"):
        """
        Summarize and visualize the results using the specified metric.

        Args:
            metric: The metric to use for comparison (default: 'balanced_accuracy')

        Produces:
        - A summary table of key metrics
        - A bar plot comparing test metrics across models and datasets
        - A heatmap for the specified metric across models and datasets
        """
        # Collect data into a DataFrame for easy visualization
        summary_data = []

        for dataset_name, model_dict in self.results.items():
            for model_name, strategy_dict in model_dict.items():
                for strategy, mech_dict in strategy_dict.items():
                    for mechanism, metrics in mech_dict.items():
                        if "test_metrics" not in metrics:
                            continue
                        test_metrics = metrics["test_metrics"]
                        summary_data.append({
                            "Dataset": dataset_name,
                            "Model": model_name,
                            "Strategy": strategy,
                            "Mechanism": mechanism,
                            "Accuracy": test_metrics.get("accuracy", 0),
                            "Balanced Accuracy": test_metrics.get("balanced_accuracy", 0),
                            "Precision": test_metrics.get("precision", 0),
                            "Recall": test_metrics.get("recall", 0),
                            "F1-Score": test_metrics.get("f1_score", 0)
                        })

        df = pd.DataFrame(summary_data)

        if df.empty:
            print("No results to summarize.")
            return

        print("Summary of Results:")
        print(df)

        # Plot comparison
        plt.figure(figsize=(14, 8))
        column_name = metric.replace("_", " ").title()



        if column_name not in df.columns:
            column_name = "Balanced Accuracy"

        sns.barplot(data=df, x="Dataset", y=column_name, hue="Model", errorbar=None)
        plt.title(f"Test {column_name} Comparison Across Models and Datasets")
        plt.ylabel(column_name)
        plt.xlabel("Dataset")
        plt.legend(loc="lower right")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.results_dir / f"test_{metric}_comparison.png")
        plt.close()

        # Heatmap (if multiple datasets and models)
        if len(df["Dataset"].unique()) > 1 and len(df["Model"].unique()) > 1:
            df["Model+Strategy"] = df["Model"] + "_" + df["Mechanism"]
            pivot_df = df.pivot(index="Model+Strategy", columns="Dataset", values=column_name)
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_df, annot=True, cmap="coolwarm", fmt=".3f", vmin=0, vmax=1)
            plt.title(f"{column_name} Heatmap")
            plt.tight_layout()
            plt.savefig(self.results_dir / f"{metric}_heatmap.png")
            plt.close()

    def plot_confusion_matrices(self, model_name: str):
        for dataset_name in self.loader.list_datasets():
            if dataset_name not in self.results or model_name not in self.results[dataset_name]:
                continue
            for dim_reduction, mech_dict in self.results[dataset_name][model_name].items():
                for feature_extraction in mech_dict:
                    self.plot_confusion_matrix_from_results(dataset_name, model_name, dim_reduction, feature_extraction)

    def plot_confusion_matrix_from_results(self, dataset_name: str, model_name: str,
                                           dim_reduction: str = "raw", feature_extraction: str = "raw", normalized=True):
        metrics = self.results[dataset_name][model_name][dim_reduction][feature_extraction]
        class_names = self.loader.get_dataset_class_names(dataset_name)

        if normalized:
            conf_matrix = metrics.get("normalized_confusion_matrix")
            fmt = ".2f"
            title_prefix = "Normalized "
        else:
            conf_matrix = metrics.get("confusion_matrix")
            fmt = "d"
            title_prefix = ""

        if conf_matrix is None:
            return

        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title(f"{title_prefix}Confusion Matrix: {model_name} ({feature_extraction}) on {dataset_name}")
        plt.tight_layout()

        output_path = self.results_dir.joinpath(dataset_name, model_name, dim_reduction, feature_extraction)
        output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path / f"{'normalized_' if normalized else ''}confusion_matrix.png")
        plt.close()

    def compare_models(self, dataset: str, metric="balanced_accuracy"):
        if dataset not in self.results:
            print(f"No results for dataset: {dataset}")
            return

        model_scores = []
        for model_name, strategy_dict in self.results[dataset].items():
            for strategy, mech_dict in strategy_dict.items():
                for mechanism, metrics in mech_dict.items():
                    if "test_metrics" not in metrics:
                        continue
                    validation_metric = metrics["validation_metrics"].get(metric,
                                                                          metrics["validation_metrics"].get("accuracy",
                                                                                                            0))
                    test_metric = metrics["test_metrics"].get(metric, metrics["test_metrics"].get("accuracy", 0))
                    model_scores.append({
                        "Model": model_name,
                        "Strategy": strategy,
                        "Mechanism": mechanism,
                        f"Validation {metric}": validation_metric,
                        f"Test {metric}": test_metric
                    })

        if not model_scores:
            print(f"No model results available for dataset: {dataset}")
            return

        df = pd.DataFrame(model_scores)
        df["Model_Mech"] = df["Model"] + "_" + df["Mechanism"]
        df = df.sort_values(by=f"Test {metric}", ascending=False)

        print(f"\nComparison for {dataset} using {metric}:\n")
        print(df)

        # Plot
        plt.figure(figsize=(12, 6))
        metric_columns = ["Validation f1_weighted", "Test f1_weighted"]
        df_filtered = df[["Model_Mech"] + metric_columns]

        df_melted = df_filtered.melt(id_vars=["Model_Mech"], var_name="Metric Type", value_name="Score")
        df_melted["Score"] = pd.to_numeric(df_melted["Score"], errors="coerce")

        with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None):  # more options can be specified also
            print(df_melted)

        sns.barplot(data=df_melted, x="Model_Mech", y="Score", hue="Metric Type", errorbar=None)
        plt.title(f"Model Comparison for {dataset} - {metric}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        output_path = self.results_dir / dataset
        output_path.mkdir(exist_ok=True)
        plt.savefig(output_path / f"model_comparison_{metric}.png")
        plt.close()

        return df

    def compare_datasets(self, model_name: str, metric="balanced_accuracy"):
        if model_name not in self.list_models():
            print(f"No such model: {model_name}")
            return

        dataset_scores = []
        for dataset, model_dict in self.results.items():
            if model_name not in model_dict:
                continue
            for strategy, mech_dict in model_dict[model_name].items():
                for mechanism, metrics in mech_dict.items():
                    if "test_metrics" not in metrics:
                        continue
                    val = metrics["validation_metrics"].get(metric, metrics["validation_metrics"].get("accuracy", 0))
                    test = metrics["test_metrics"].get(metric, metrics["test_metrics"].get("accuracy", 0))
                    dataset_scores.append({
                        "Dataset": dataset,
                        "Strategy": strategy,
                        "Mechanism": mechanism,
                        f"Validation {metric}": val,
                        f"Test {metric}": test
                    })

        if not dataset_scores:
            print(f"No dataset results available for model: {model_name}")
            return

        df = pd.DataFrame(dataset_scores)
        df["Dataset_Mech"] = df["Dataset"] + "_" + df["Mechanism"]
        df = df.sort_values(by=f"Test {metric}", ascending=False)

        print(f"\nComparison for model {model_name} using {metric}:\n")
        print(df)

        metric_columns = ["Validation f1_weighted", "Test f1_weighted"]
        df_filtered = df[["Dataset_Mech"] + metric_columns]

        df_melted = df_filtered.melt(id_vars=["Dataset_Mech"], var_name="Metric Type", value_name="Score")
        df_melted["Score"] = pd.to_numeric(df_melted["Score"], errors="coerce")

        with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None):  # more options can be specified also
            print(df_melted)

        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_melted, x="Dataset_Mech", y="Score", hue="Metric Type", errorbar=None)
        plt.title(f"Dataset Comparison for {model_name} - {metric}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        output_path = self.results_dir / "model_comparisons"
        output_path.mkdir(exist_ok=True)
        plt.savefig(output_path / f"{model_name}_dataset_comparison_{metric}.png")
        plt.close()

        return df

    def _get_param_grid_for_model(self, model_name: str) -> Dict[str, List[Any]]:
        """
        Get the parameter grid for grid search based on the model name.

        Args:
            model_name: Name of the model

        Returns:
            Parameter grid dictionary
        """
        # Define parameter grids for different models
        param_grids = {
            "Rf": {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [10, 20, 30, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "bootstrap": [True, False]
            },
            "Svm": {
                "C": [0.1, 1, 10, 100],
                "gamma": ["scale", 1e-3, 1e-4],
                "kernel": ["rbf", "linear"]
            },
            "Lda": {
                "solver": ["svd", "lsqr", "eigen"],
                "shrinkage": [None, "auto", 0.1, 0.5, 0.9]
            },
            "Qda": {
                "reg_param": [0.0, 0.1, 0.5],
                "store_covariance": [True, False]
            },
            "Gnb": {
                "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]
            },
            "Lr": {
                "C": [0.01, 0.1, 1, 10, 100],
                "solver": ["liblinear", "saga"],
                "penalty": ["l1", "l2"]
            },
            "Dtc": {
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "criterion": ["gini", "entropy"]
            },
            "Ab": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "estimator__max_depth": [1, 3, 5]
            }
        }

        if model_name not in param_grids.keys():
            raise Exception(f"Model {model_name} not found in param grid")

        # Return the param grid for the requested model or an empty dict if not found
        return param_grids.get(model_name, {})


if __name__ == "__main__":
    # Select optimization metric
    optimization_metric = "f1_weighted"
    import matplotlib
    # Use non-interactive plot
    matplotlib.use('Agg')

    # Initialize the pipeline with the DataLoader and ImageProcessor
    loader = DatasetLoader()
    loader.load_datasets(split=True)
    pipeline = McMls(results_dir="results", optimize_metric=optimization_metric)
    pipeline.loader = loader

    # Define your feature strategies
    strategies = [
        OpOption("raw", "raw", enabled=False),   # Too much time required
        OpOption("raw", "HOG", enabled=True),    # Too much time required
        OpOption("raw", "GLCM", enabled=True),
        OpOption("raw", "SIFT", enabled=False),  # Variable-length — not ready unless you BoW
        OpOption("raw", "ResNet", enabled=True),
        OpOption("raw", "Hist", enabled=False),  # Really bad results

        OpOption("PCA", "raw", enabled=True),
        OpOption("PCA", "GLCM", enabled=False),
        OpOption("PCA", "Hist", enabled=False),
        OpOption("PCA", "HOG", enabled=True),  # Only if HOG vectors are high-dimensional
        OpOption("PCA", "SIFT", enabled=True),
        OpOption("PCA", "ResNet", enabled=True),
    ]

    mut = ModelFactory.list_all()
    print(mut)
    # Run full experiment suite with raw + feature-based workflows
    save = False
    for option in strategies:
        mut2 = mut
        if option.enabled:
            if option.dim_reduction == "raw":
                if "Ab" in mut2:
                    mut2.remove("Ab")
            if option.dim_reduction == "raw" and option.feature_extraction == "ResNet":
                if "Lr" in mut2:
                    mut2.remove("Lr")
            save = True
            pipeline.evaluate_subset(model_names=mut2, dataset_names=pipeline.loader.list_datasets(), option=option)

    # Save results to disk
    if save:
        pipeline.save_results()

    # Summarize results using balanced accuracy
    pipeline.summarize_results(metric=optimization_metric)

    # Compare models for each dataset
    for d in loader.list_datasets():
        pipeline.compare_models(d, metric=optimization_metric)

    # Compare datasets for each model
    for model in pipeline.list_models():
        pipeline.compare_datasets(model, metric=optimization_metric)

    # Plot confusion matrices for each model
    for model in pipeline.list_models():
        pipeline.plot_confusion_matrices(model)
