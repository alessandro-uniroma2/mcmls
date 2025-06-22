import json
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
from hashlib import sha256
from pathlib import Path
from typing import Union, Dict, List, Tuple, Any

import keras
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from keras.src.activations import gelu
from keras.src.callbacks import History
from numpy import mean, std
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import make_scorer, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from core.features import OpOption, FeatureProcessor

DEFAULT_EPOCH = 80


class IModel(ABC):

    def __init__(self, **kwargs) -> None:
        self.checkpoint_filepath = tempfile.NamedTemporaryFile(delete=False, suffix=".weights.h5").name
        self.kwargs = kwargs
        self.train = self.kwargs.pop("training", None)  # Remove DirectoryIterator because it can't be serialized
        self.val = kwargs.get("validation")
        self.history = None
        self.number_of_classes = kwargs.get("number_of_classes")
        self.regularize = kwargs.get("regularize", False)
        self.patience = kwargs.get("patience", 10)
        self.num_epoch = kwargs.get("num_epoch", DEFAULT_EPOCH)
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.weight_decay = kwargs.get("weight_decay", 0.0001)
        self.activation = kwargs.get("activation", gelu)
        self.input_shape = kwargs.get("input_shape")
        self.batch_size = 32
        self.callbacks = []
        self.option: Union[None, OpOption] = kwargs.get("option", None)
        self.preprocessor: Union[None, FeatureProcessor] = kwargs.get("preprocessor", None)

        # Because we all know the ultimate answer is 42
        self.random_state = 42

        # Params really needed in classic learning
        self.solver = kwargs.get("solver")
        self.n_jobs = kwargs.get("n_jobs", -1)
        self.multi_class = kwargs.get("multi_class")
        self.cv = kwargs.get("cv", RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1))

        # Inner model supported by scikit
        self.model = None

        # Storage for evaluation results
        self.evaluation_results = {}

    @abstractmethod
    def create_model(self, preprocess=True):
        pass

    def preprocess(self, X, training=False):
        if hasattr(self, "preprocessor") and self.preprocessor:
            if training:
                return self.preprocessor.fit_transform(X)
            else:
                return self.preprocessor.transform(X)
        return X

        # Training the model with Early Stopping Criterion
        # on Validation Loss.
    def fit(self, X, y):
        X = self.preprocess(X, training=True)
        x_train_balanced, y_train_balanced = self.smote(X, y)
        self.model.fit(x_train_balanced, y_train_balanced)

    def smote(self, x_train, y_train) -> tuple:
        smote = SMOTE(random_state=self.random_state)
        x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)
        return x_train_balanced, y_train_balanced

    def load_kwargs(self, **kwargs):
        self.kwargs = kwargs
        self.train = kwargs.get("training")
        self.val = kwargs.get("validation")
        self.number_of_classes = kwargs.get("number_of_classes")
        self.regularize = kwargs.get("regularize", False)
        self.patience = kwargs.get("patience", 10)
        self.num_epoch = kwargs.get("num_epoch", DEFAULT_EPOCH)
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.weight_decay = kwargs.get("weight_decay", 0.0001)
        self.activation = kwargs.get("activation", gelu)
        self.input_shape = kwargs.get("input_shape")

    def enable_checkpoint(self):
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            self.checkpoint_filepath,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
        )
        self.callbacks.append(checkpoint_callback)

    def evaluate(self, X, y, dataset_name="default", metrics=None):
        """
        Evaluate the model using cross-validation and store results

        Args:
            X: Feature data
            y: Target data
            dataset_name: Name of the dataset for tracking results
            metrics: List of scoring metrics to evaluate
                If None, uses ['balanced_accuracy', 'f1_weighted', 'roc_auc_ovr_weighted']

        Returns:
            Dict with evaluation metrics
        """
        # Create the model if it doesn't exist
        if self.model is None:
            self.create_model()

        # Preprocess the data
        X_processed = self.preprocess(X, training=True)

        # Default metrics if None provided
        if metrics is None:
            # Choose metrics based on problem type (binary vs multi-class)
            if len(np.unique(y)) == 2:
                metrics = ['balanced_accuracy', 'f1', 'roc_auc', 'precision', 'recall']
            else:
                metrics = ['balanced_accuracy', 'f1_weighted', 'f1_macro', 'roc_auc_ovr_weighted']

        # Evaluate with multiple metrics
        results = {
            'metrics': {}
        }

        print(f'Dataset: {dataset_name}')

        for metric in metrics:
            try:
                scores = cross_val_score(self.model, X_processed, y, scoring=metric, cv=self.cv, n_jobs=self.n_jobs)
                results['metrics'][metric] = {
                    'scores': scores,
                    'mean': mean(scores),
                    'std': std(scores)
                }
                print(f'Mean {metric}: {mean(scores):.3f} (±{std(scores):.3f})')
            except Exception as e:
                print(f"Error calculating {metric}: {str(e)}")

        # For backward compatibility, still calculate accuracy
        accuracy_scores = cross_val_score(self.model, X_processed, y, scoring='accuracy', cv=self.cv,
                                          n_jobs=self.n_jobs)
        results['cv_scores'] = accuracy_scores
        results['mean_accuracy'] = mean(accuracy_scores)
        results['std_accuracy'] = std(accuracy_scores)

        # Store in the evaluation results dictionary
        if dataset_name not in self.evaluation_results:
            self.evaluation_results[dataset_name] = {}

        self.evaluation_results[dataset_name]['baseline'] = results

        return results

    def grid_search(self, X, y, param_grid, dataset_name="malimg", plot=True, output_dir="./",
                    scoring='balanced_accuracy', search_strategy="random"):
        """
        Perform grid search CV to find optimal parameters

        Args:
            X: Feature data
            y: Target data
            param_grid: Dictionary of parameters to search
            dataset_name: Name of the dataset for tracking results
            plot: Whether to create visualization plots
            output_dir: Directory to save plots
            scoring: Scoring metric to use (default: 'balanced_accuracy')
                Options include: 'balanced_accuracy', 'f1', 'f1_weighted',
                'f1_macro', 'roc_auc', 'roc_auc_ovr', 'precision', 'recall',
                'average_precision', etc.
            search_strategy: Search strategy for best CV. Can be "grid" or "random"

        Returns:
            Best parameters and model
        """
        # Create the model if it doesn't exist
        if self.model is None:
            self.create_model()

        # Preprocess the data
        X_processed = self.preprocess(X, training=True)

        # Create grid search
        if search_strategy == "grid":
            search_func = "GridSearchCV"
            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                cv=self.cv,
                n_jobs=self.n_jobs,
                scoring=scoring,
                return_train_score=True,
                verbose=1
            )
        else:
            search_func = "RandomizedSearchCV"
            grid_search = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=param_grid,
                n_iter=5,  # You can adjust this — fewer iterations = faster
                cv=self.cv,
                n_jobs=self.n_jobs,
                scoring=scoring,
                return_train_score=True,
                verbose=0,
                random_state=self.random_state
            )

        # Run grid search
        print(f"Running {search_func} for dataset: {dataset_name}")
        grid_search.fit(X_processed, y)

        # Get results as DataFrame
        results_df = pd.DataFrame(grid_search.cv_results_)

        # Report best parameters and score
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

        # Store in the evaluation results dictionary
        if dataset_name not in self.evaluation_results:
            self.evaluation_results[dataset_name] = {}

        self.evaluation_results[dataset_name]['grid_search'] = {
            'grid_search': grid_search,
            'results_df': results_df,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }

        # Create visualization if requested
        if plot:
            self._plot_grid_search_results(dataset_name, output_dir)

        # Update model with best parameters
        self.model.set_params(**grid_search.best_params_)

        return grid_search.best_params_, self.model

    def _plot_grid_search_results(self, dataset_name, output_dir="."):
        """
        Visualize the grid search results

        Args:
            dataset_name: Name of the dataset
            output_dir: Directory to save the plot
        """
        # Make sure the output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Get results
        if dataset_name not in self.evaluation_results or 'grid_search' not in self.evaluation_results[dataset_name]:
            print(f"No grid search results found for dataset: {dataset_name}")
            return

        results = self.evaluation_results[dataset_name]['grid_search']['results_df']

        # Get the parameter with most variation
        param_names = [p for p in results.columns if p.startswith('param_')]
        if not param_names:
            print("No parameters found in grid search results")
            return

        plt.figure(figsize=(15, 6))

        # Plot for each varying parameter
        plots_created = 0
        for i, param in enumerate(param_names):
            param_values = results[param].astype(str)
            unique_values = param_values.unique()

            if len(unique_values) > 1:
                plt.subplot(1, len(param_names), i + 1)
                plots_created += 1

                # Group by parameter value and get mean scores
                grouped = results.groupby(param)['mean_test_score'].mean()

                # Sort values if numeric
                try:
                    param_numeric = pd.to_numeric(results[param])
                    sorted_idx = np.argsort(param_numeric.unique())
                    unique_values = param_numeric.unique().tolist()
                    x_values = [unique_values[i] for i in sorted_idx]
                    y_values = [grouped[unique_values[i]] for i in sorted_idx]
                except:
                    unique_values = list(map(str, grouped.index))
                    grouped.index = grouped.index.map(str)
                    x_values = unique_values
                    y_values = [grouped[v] for v in unique_values]

                plt.plot(x_values, y_values, 'o-', label='Mean CV Score')
                plt.title(f'Effect of {param.replace("param_", "")}')
                plt.xlabel(param.replace("param_", ""))
                plt.ylabel('Mean CV Score')
                plt.grid(True)

        if plots_created > 0:
            plt.tight_layout()
            image_path = Path(output_dir).joinpath(f"{dataset_name}_{self.option}_gridsearch_results.png")
            plt.savefig(f"{image_path}")
            plt.close()
            print(f"Grid search results plot saved to {image_path}")
        else:
            plt.close()
            print("No plots created - parameters may not have multiple values")

    def plot_learning_curve(self, X, y, dataset_name="malimg", output_dir="."):
        """
        Plot learning curve for the model

        Args:
            X: Feature data
            y: Target data
            dataset_name: Name of the dataset
            output_dir: Directory to save the plot
        """
        # Make sure the output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Create the model if it doesn't exist
        if self.model is None:
            self.create_model()

        # Preprocess the data
        X_processed = self.preprocess(X, training=True)

        # Calculate learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            self.model,
            X_processed,
            y,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=self.cv,
            scoring='balanced_accuracy',
            n_jobs=self.n_jobs
        )

        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Plot learning curve
        plt.figure(figsize=(10, 6))
        plt.title(f'Learning Curve - {dataset_name}')
        plt.xlabel('Training examples')
        plt.ylabel('Score')
        plt.grid()

        plt.fill_between(train_sizes, train_mean - train_std,
                         train_mean + train_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_mean - test_std,
                         test_mean + test_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
        plt.legend(loc="best")

        image_path = Path(output_dir).joinpath(f"{dataset_name}_{self.option}_learning_curve.png")
        plt.savefig(f"{image_path}")
        plt.close()
        print(f"Learning curve plot saved to {image_path}")

        # Store results
        if dataset_name not in self.evaluation_results:
            self.evaluation_results[dataset_name] = {}

        self.evaluation_results[dataset_name]['learning_curve'] = {
            'train_sizes': train_sizes,
            'train_scores': train_scores,
            'test_scores': test_scores,
            'train_mean': train_mean,
            'train_std': train_std,
            'test_mean': test_mean,
            'test_std': test_std,
            'op_option': self.option.to_dict()
        }

    def evaluate_test_set(self, X_test, y_test, dataset_name="malimg", output_dir="./"):
        """
        Evaluate model on test set and create confusion matrix

        Args:
            X_test: Test feature data
            y_test: Test target data
            dataset_name: Name of the dataset
            output_dir: Directory to save the plots

        Returns:
            Dict with test evaluation metrics
        """
        # Make sure the output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Create the model if it doesn't exist
        if self.model is None:
            self.create_model()

        # Preprocess the data
        X_processed = self.preprocess(X_test, training=False)

        # Make predictions
        y_pred = self.model.predict(X_processed)

        # For models that can predict probabilities
        if hasattr(self.model, "predict_proba"):
            try:
                y_proba = self.model.predict_proba(X_processed)
            except:
                y_proba = None
        else:
            y_proba = None

        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_normalized = confusion_matrix(y_test, y_pred, normalize='true')

        # Plot standard confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {dataset_name} - {self.option}')
        plt.xlabel('Predicted')
        plt.ylabel('True')

        image_path = Path(output_dir).joinpath(f"{dataset_name}_{self.option}_confusion_matrix.png")
        plt.savefig(f"{image_path}")
        plt.close()

        # Plot normalized confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')
        plt.title(f'Normalized Confusion Matrix - {dataset_name} - {self.option}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        image_path = Path(output_dir).joinpath(f"{dataset_name}_{self.option}_normalized_confusion_matrix.png")
        plt.savefig(f"{image_path}")
        plt.close()

        print(f"Confusion matrices saved to {image_path}")

        # Get classification report
        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
        print(f"Classification Report - {dataset_name} - {self.option}")
        print(classification_report(y_test, y_pred, zero_division=1))

        # Store basic results
        test_results = {
            'confusion_matrix': cm,
            'normalized_confusion_matrix': cm_normalized,
            'classification_report': class_report,
            'accuracy': class_report['accuracy']
        }

        # If we have probability predictions and it's a binary classification
        if y_proba is not None and len(np.unique(y_test)) == 2:
            from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

            # Calculate ROC curve and ROC area
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)

            # Plot ROC curve
            plt.figure(figsize=(8, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {dataset_name}')
            plt.legend(loc="lower right")
            image_path = Path(output_dir).joinpath(f"{dataset_name}_{self.option}_roc_curve.png")
            plt.savefig(f"{image_path}")
            plt.close()

            # Calculate precision-recall curve
            precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])
            avg_precision = average_precision_score(y_test, y_proba[:, 1])

            # Plot precision-recall curve
            plt.figure(figsize=(8, 8))
            plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title(f'Precision-Recall Curve - {dataset_name}')
            plt.legend(loc="lower left")
            image_path = Path(output_dir).joinpath(f"{dataset_name}_{self.option}_precision_recall_curve.png")
            plt.savefig(f"{image_path}")
            plt.close()

            # Add to results
            test_results.update({
                'roc_auc': roc_auc,
                'average_precision': avg_precision,
                'fpr': fpr,
                'tpr': tpr,
                'precision': precision,
                'recall': recall
            })

            print(f"ROC AUC: {roc_auc:.4f}")
            print(f"Average Precision: {avg_precision:.4f}")

        # For multiclass problems, plot per-class metrics
        if len(np.unique(y_test)) > 2:
            # Extract per-class metrics from the classification report
            classes = [key for key in class_report.keys() if key not in ['accuracy', 'macro avg', 'weighted avg']]

            # Plot F1, precision, and recall for each class
            metrics = ['precision', 'recall', 'f1-score']
            plt.figure(figsize=(12, 6))

            x = np.arange(len(classes))
            width = 0.25

            for i, metric in enumerate(metrics):
                values = [class_report[cls][metric] for cls in classes]
                plt.bar(x + (i - 1) * width, values, width, label=metric)

            plt.xlabel('Classes')
            plt.ylabel('Score')
            plt.title(f'Per-Class Performance Metrics - {dataset_name} - {self.option}')
            plt.xticks(x, classes)
            plt.legend()
            image_path = Path(output_dir).joinpath(f"{dataset_name}_{self.option}_per_class_metrics.png")
            plt.savefig(f"{image_path}")
            plt.close()

            print(f"Per-class metrics plot saved to {image_path}")

        if dataset_name not in self.evaluation_results:
            self.evaluation_results[dataset_name] = {}

        self.evaluation_results[dataset_name]['test_evaluation'] = test_results

        return test_results

    def compare_datasets(self, output_dir="./", metric='balanced_accuracy'):
        """
        Compare performance across datasets

        Args:
            output_dir: Directory to save the plot
            metric: The metric to use for comparison
                Default is 'balanced_accuracy'
        """
        # Make sure the output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Check if we have evaluation results
        if not self.evaluation_results:
            print("No evaluation results available for comparison")
            return

        # Collect metric values for each dataset
        datasets = []
        baseline_scores = []
        best_cv_scores = []
        test_scores = []

        for dataset_name, results in self.evaluation_results.items():
            datasets.append(dataset_name)

            # Baseline score - try to get the specified metric first
            if 'baseline' in results:
                if 'metrics' in results['baseline'] and metric in results['baseline']['metrics']:
                    baseline_scores.append(results['baseline']['metrics'][metric]['mean'])
                else:
                    # Fallback to accuracy for backward compatibility
                    baseline_scores.append(results['baseline']['mean_accuracy'])
            else:
                baseline_scores.append(None)

            # Best CV score from grid search
            if 'grid_search' in results:
                best_cv_scores.append(results['grid_search']['best_score'])
            else:
                best_cv_scores.append(None)

            # Test score
            if 'test_evaluation' in results:
                if metric == 'balanced_accuracy' and 'accuracy' in results['test_evaluation']:
                    # Use regular accuracy if balanced not available
                    test_scores.append(results['test_evaluation']['accuracy'])
                elif metric in results['test_evaluation']['classification_report']:
                    test_scores.append(results['test_evaluation']['classification_report'][metric])
                else:
                    # Fallback to accuracy
                    test_scores.append(results['test_evaluation']['accuracy'])
            else:
                test_scores.append(None)

        # Create comparison plot
        plt.figure(figsize=(12, 6))
        x = np.arange(len(datasets))
        width = 0.25

        # Plot bars
        baseline_bars = plt.bar(x - width, baseline_scores, width, label='Baseline CV')
        best_cv_bars = plt.bar(x, best_cv_scores, width, label='Best CV (GridSearch)')
        test_bars = plt.bar(x + width, test_scores, width, label='Test Set')

        # Add labels and title
        plt.xlabel('Datasets')
        plt.ylabel('Accuracy')
        plt.title('Model Performance Comparison Across Datasets')
        plt.xticks(x, datasets)
        plt.legend()

        # Add values on top of bars
        def add_labels(bars):
            for bar in bars:
                if bar.get_height() is not None:
                    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                             f'{bar.get_height():.3f}', ha='center', va='bottom')

        add_labels(baseline_bars)
        add_labels(best_cv_bars)
        add_labels(test_bars)

        plt.tight_layout()
        image_path = Path(output_dir).joinpath(f"{self.option}_dataset_comparison.png")
        plt.savefig(f"{image_path}")
        plt.close()
        print(f"Dataset comparison plot saved to {image_path}")

    def create_multi_metric_pipeline(self, steps=None, param_grid=None, custom_scorers=None):
        """
        Create a pipeline with multiple metrics for evaluation

        Args:
            steps: List of (name, transform) tuples for the pipeline steps
                  If None, uses a single step with the model
            param_grid: Parameter grid for the pipeline
                       If None, returns a pipeline without grid search
            custom_scorers: Dictionary of {name: scorer} for custom metrics
                           If None, uses default multi-metric scorers

        Returns:
            Pipeline or GridSearchCV object with multiple metrics
        """
        # Create default scorers if none provided
        if custom_scorers is None:
            custom_scorers = {
                'balanced_accuracy': make_scorer(balanced_accuracy_score),
                'f1_weighted': make_scorer(f1_score, average='weighted'),
                'f1_macro': make_scorer(f1_score, average='macro'),
                'precision_macro': make_scorer(precision_score, average='macro'),
                'recall_macro': make_scorer(recall_score, average='macro')
            }

        # Create the model if it doesn't exist
        if self.model is None:
            self.create_model()

        # Create pipeline steps
        if steps is None:
            # Use a single step with the model
            pipe_steps = [('model', self.model)]
        else:
            pipe_steps = steps

        # Create pipeline
        pipeline = Pipeline(pipe_steps)

        # If no param_grid, return the pipeline
        if param_grid is None:
            return pipeline, custom_scorers

        # Create grid search with multiple metrics
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring=custom_scorers,
            refit='balanced_accuracy',  # Primary metric for selecting best model
            cv=self.cv,
            n_jobs=self.n_jobs,
            return_train_score=True,
            verbose=1
        )

        return grid_search, custom_scorers

    def evaluate_pipeline(self, pipeline, X, y, metrics, dataset_name="malimg"):
        """
        Evaluate a pipeline with custom metrics

        Args:
            pipeline: Pipeline or estimator to evaluate
            X: Feature data
            y: Target data
            metrics: Dictionary of {name: scorer} for metrics
            dataset_name: Name of the dataset for tracking results

        Returns:
            Dictionary of evaluation results
        """
        # Preprocess the data
        X_processed = self.preprocess(X, training=True)

        # Evaluate with each metric
        results = {
            'metrics': {}
        }

        print(f'Dataset: {dataset_name} - Opt: {self.option} - Pipeline Evaluation')

        for metric_name, scorer in metrics.items():
            try:
                # For custom scorers, we need to use the scorer directly
                scores = cross_val_score(
                    pipeline, X_processed, y,
                    scoring=scorer,
                    cv=self.cv,
                    n_jobs=self.n_jobs
                )

                results['metrics'][metric_name] = {
                    'scores': scores,
                    'mean': mean(scores),
                    'std': std(scores)
                }
                print(f'Mean {metric_name}: {mean(scores):.3f} (±{std(scores):.3f})')

            except Exception as e:
                print(f"Error calculating {metric_name}: {str(e)}")

        # Store in the evaluation results dictionary
        if dataset_name not in self.evaluation_results:
            self.evaluation_results[dataset_name] = {}

        self.evaluation_results[dataset_name]['pipeline'] = results

        return results

    def fit_evaluate_pipeline(self, grid_search, X_train, y_train, X_test, y_test, dataset_name="malimg",
                              output_dir="./"):
        """
        Fit a pipeline with grid search and evaluate on test data

        Args:
            grid_search: GridSearchCV object with pipeline
            X_train: Training feature data
            y_train: Training target data
            X_test: Test feature data
            y_test: Test target data
            dataset_name: Name of the dataset
            output_dir: Directory to save results
        Returns:
            Dictionary with results
        """
        # Make sure the output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Preprocess the data
        X_train_processed = self.preprocess(X_train, training=True)
        X_test_processed = self.preprocess(X_test, training=False)

        # Fit the grid search
        print(f"Fitting grid search pipeline for dataset: {dataset_name} - {self.option}")
        grid_search.fit(X_train_processed, y_train)

        # Get results as DataFrame
        results_df = pd.DataFrame(grid_search.cv_results_)

        # Get best estimator and parameters
        best_estimator = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        # Report best parameters and score
        print(f"Best parameters: {best_params}")
        print(f"Best cross-validation score: {best_score:.3f}")

        # Visualize grid search results
        self._plot_grid_search_pipeline_results(results_df, dataset_name, output_dir)

        # Evaluate on test set
        y_pred = best_estimator.predict(X_test_processed)

        # Calculate metrics on test set
        cm = confusion_matrix(y_test, y_pred)
        cm_normalized = confusion_matrix(y_test, y_pred, normalize='true')
        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)

        # Plot confusion matrices
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Pipeline Confusion Matrix - {dataset_name} - {self.option}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        image_path = Path(output_dir).joinpath(f"{dataset_name}_{self.option}_pipeline_confusion_matrix.png")
        plt.savefig(f"{image_path}")
        plt.close()

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')
        plt.title(f'Pipeline Normalized Confusion Matrix - {dataset_name} - {self.option}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        image_path = Path(output_dir).joinpath(f"{dataset_name}_{self.option}_pipeline_normalized_confusion_matrix.png")
        plt.savefig(f"{image_path}")
        plt.close()

        # Print classification report
        print(f"Pipeline Classification Report - {dataset_name} - {self.option}")
        print(classification_report(y_test, y_pred, zero_division=1))

        # Per-class metrics visualization
        if len(np.unique(y_test)) > 2:
            classes = [key for key in class_report.keys() if key not in ['accuracy', 'macro avg', 'weighted avg']]
            metrics = ['precision', 'recall', 'f1-score']
            plt.figure(figsize=(12, 6))

            x = np.arange(len(classes))
            width = 0.25

            for i, metric in enumerate(metrics):
                values = [class_report[cls][metric] for cls in classes]
                plt.bar(x + (i - 1) * width, values, width, label=metric)

            plt.xlabel('Classes')
            plt.ylabel('Score')
            plt.title(f'Pipeline Per-Class Performance - {dataset_name} - {self.option}')
            plt.xticks(x, classes)
            plt.legend()
            image_path = Path(output_dir).joinpath(
                f"{dataset_name}_{self.option}_pipeline_per_class_metrics.png")
            plt.savefig(f"{image_path}")
            plt.close()

        # Store results
        pipeline_results = {
            'best_estimator': best_estimator,
            'best_params': best_params,
            'best_score': best_score,
            'confusion_matrix': cm,
            'normalized_confusion_matrix': cm_normalized,
            'classification_report': class_report,
            'op_option': self.option
        }

        if dataset_name not in self.evaluation_results:
            self.evaluation_results[dataset_name] = {}

        self.evaluation_results[dataset_name]['pipeline_evaluation'] = pipeline_results

        return pipeline_results

    def _plot_grid_search_pipeline_results(self, results_df, dataset_name, output_dir="./"):
        """
        Plot grid search results for pipeline

        Args:
            results_df: DataFrame with grid search results
            dataset_name: Name of the dataset
            output_dir: Directory to save plots
        """
        # Extract parameter columns
        param_cols = [c for c in results_df.columns if c.startswith('param_')]
        if not param_cols:
            print("No parameters found in grid search results")
            return

        # Extract score columns for different metrics
        score_cols = [c for c in results_df.columns if c.startswith('mean_test_')]
        metrics = [s.replace('mean_test_', '') for s in score_cols]

        # Plot for each parameter
        for param in param_cols:
            param_name = param.replace('param_', '')
            param_values = results_df[param].astype(str)
            unique_values = param_values.unique()

            if len(unique_values) <= 1:
                continue

            plt.figure(figsize=(10, 6))

            # Try to convert to numeric for better plotting
            try:
                numeric_values = pd.to_numeric(results_df[param])
                has_numeric = True
            except:
                has_numeric = False

            # Plot for each metric
            for metric, score_col in zip(metrics, score_cols):
                if has_numeric:
                    # For numeric parameters, do a line plot
                    grouped = results_df.groupby(param)[score_col].mean()
                    sorted_idx = np.argsort(pd.to_numeric(grouped.index))
                    plt.plot(
                        [grouped.index[i] for i in sorted_idx],
                        [grouped.iloc[i] for i in sorted_idx],
                        'o-',
                        label=metric
                    )
                else:
                    # For categorical parameters, do a bar plot
                    grouped = results_df.groupby(param)[score_col].mean()
                    plt.bar(
                        np.arange(len(unique_values)) + (metrics.index(metric) - len(metrics) / 2) * 0.15,
                        [grouped[v] for v in unique_values],
                        width=0.15,
                        label=metric
                    )
                    plt.xticks(np.arange(len(unique_values)), unique_values)

            plt.title(f'Effect of {param_name} on Different Metrics - {dataset_name} - {self.option}')
            plt.xlabel(param_name)
            plt.ylabel('Score')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            image_path = Path(output_dir).joinpath(
                f"{dataset_name}_{self.option}_{param_name}_metrics.png")
            plt.savefig(f"{image_path}")
            plt.close()

    def predict(self, X):
        """
        Make predictions on new data

        Args:
            X: Input features

        Returns:
            Predicted class labels
        """
        X = self.preprocess(X, training=False)
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities if the model supports it

        Args:
            X: Input features

        Returns:
            Class probabilities or None if not supported
        """
        if hasattr(self.model, "predict_proba"):
            X = self.preprocess(X, training=False)
            return self.model.predict_proba(X)
        return None

    def save_evaluation_results(self, path: Path):
        """
        Save evaluation results to a file

        Args:
            path: Path to save the evaluation results
        """
        # Convert evaluation results to a serializable format
        serializable_results = {}

        for dataset_name, results in self.evaluation_results.items():
            serializable_results[dataset_name] = {}

            for eval_type, eval_data in results.items():
                serializable_data = {}

                # Handle different types of evaluation data
                if eval_type == 'baseline':
                    serializable_data = {
                        'mean_accuracy': float(eval_data.get('mean_accuracy', 0)),
                        'std_accuracy': float(eval_data.get('std_accuracy', 0))
                    }

                    if 'metrics' in eval_data:
                        serializable_data['metrics'] = {}
                        for metric, metric_data in eval_data['metrics'].items():
                            serializable_data['metrics'][metric] = {
                                'mean': float(metric_data['mean']),
                                'std': float(metric_data['std'])
                            }
                elif eval_type == 'grid_search':
                    serializable_data = {
                        'best_params': eval_data['best_params'],
                        'best_score': float(eval_data['best_score'])
                    }
                elif eval_type == 'learning_curve':
                    serializable_data = {
                        'train_mean': eval_data['train_mean'].tolist(),
                        'train_std': eval_data['train_std'].tolist(),
                        'test_mean': eval_data['test_mean'].tolist(),
                        'test_std': eval_data['test_std'].tolist()
                    }
                elif eval_type == 'test_evaluation':
                    serializable_data = {
                        'accuracy': float(eval_data['accuracy']),
                        'classification_report': eval_data['classification_report']
                    }
                elif eval_type == 'pipeline_evaluation':
                    serializable_data = {
                        'best_params': eval_data['best_params'],
                        'best_score': float(eval_data['best_score']),
                        'classification_report': eval_data['classification_report']
                    }
                else:
                    # For other types, try to serialize what we can
                    for key, value in eval_data.items():
                        if key not in ['grid_search', 'best_estimator', 'results_df']:
                            try:
                                # Check if it's a numpy array or value
                                if hasattr(value, 'tolist'):
                                    serializable_data[key] = value.tolist()
                                else:
                                    serializable_data[key] = value
                            except:
                                # Skip values that can't be serialized
                                continue

                serializable_results[dataset_name][eval_type] = serializable_data

        # Write to file
        with open(str(path), "w") as f:
            json.dump(serializable_results, f, indent=2)


