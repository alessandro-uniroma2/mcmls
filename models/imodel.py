import json
import tempfile
from abc import ABC, abstractmethod
from hashlib import sha256
from pathlib import Path
from typing import Union

import keras
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from keras.src.activations import gelu
from keras.src.callbacks import History
from numpy import mean, std
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV

DEFAULT_EPOCH = 80


class IModel(ABC):

    def __init__(self, **kwargs) -> None:
        self.checkpoint_filepath = tempfile.NamedTemporaryFile(delete=False, suffix=".weights.h5").name
        self.kwargs = kwargs
        self.train = self.kwargs.pop("training", None)  # Remove DirectoryIterator cause it can't be serialized
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

        # Because we all know the ultimate answer is 42
        self.random_state = 42

        # Params really needed in classic learning
        self.solver = kwargs.get("solver")
        self.n_jobs = kwargs.get("n_jobs", -1)
        self.multi_class = kwargs.get("multi_class")
        self.cv = kwargs.get("cv", RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1))

        # Inner model supported by scikit
        self.model = None

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

    def evaluate(self, X, y):
        # evaluate the model and collect the scores
        n_scores = cross_val_score(self.model, X, y, scoring='accuracy', cv=self.cv, n_jobs=self.n_jobs)
        # report the model performance
        print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

    def search_best_cv(self, X, y, solvers: list = None):
        if solvers is None or len(solvers) == 0:
            return
        grid = dict()
        grid['solver'] = solvers
        # define search
        search = GridSearchCV(self.model, grid, scoring='accuracy', cv=self.cv, n_jobs=self.n_jobs)
        # perform the search
        results = search.fit(X, y)
        # summarize
        print('Mean Accuracy: %.3f' % results.best_score_)
        print('Config: %s' % results.best_params_)

    @abstractmethod
    def create_model(self, preprocess=True):
        pass

    # Training the model with Early Stopping Criterion
    # on Validation Loss.
    def fit(self, X, y):
        self.model.fit(X, y)

    def smote(self, x_train, y_train) -> tuple:
        smote = SMOTE(random_state=self.random_state)
        x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)
        return x_train_balanced, y_train_balanced

    def predict(self, X):
        return self.model.predict(X)

    def save_inner_model(self, path: Path):
        self.model.save(str(path))

    def load_inner_model(self, path: Path):
        self.model = keras.models.load_model(str(path))

    def save_parameters(self, path: Path):
        with open(str(path), "w") as f:
            json.dump(self.kwargs.__dict__, f)

    def load_parameters(self, path: Path):
        with open(str(path), "r") as f:
            kwargs = json.load(f)
        self.load_kwargs(**kwargs)

    def save_history(self, path: Path):
        with open(str(path), "w") as f:
            json.dump(self.history.history, f)

    def load_history(self, path: Path):
        with open(str(path), "r") as f:
            self.history = History()
            self.history.history = json.load(f)

    # This function should be used to identify specific constructions of the ViT
    def hashed_params(self):
        return sha256(json.dumps(self.kwargs, sort_keys=True).encode('utf8')).hexdigest()
