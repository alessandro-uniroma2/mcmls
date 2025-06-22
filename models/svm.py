from sklearn.kernel_approximation import Nystroem
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from models.imodel import IModel


class SvmModel(IModel):
    def __init__(self, **kwargs):
        _kwargs = {
            "kernel": "linear",
            "C": 1.0,
            "gamma": "scale",
            "verbose": True,
            "probability": False,
            "max_iter": -1
        }
        _kwargs = {**_kwargs, **kwargs}

        self.sc = StandardScaler()
        self.ns = Nystroem(kernel='rbf', gamma=0.2, n_components=500)
        super().__init__(**_kwargs)
        self.name = "Support Vector Machine Classifier"

    def create_model(self, preprocess=False):
        self.model = SVC(**self.kwargs)

    def preprocess(self, x, training=True):
        x = super().preprocess(x, training=training)

        # Flatten if needed
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)

        if training:
            scaled = self.sc.fit_transform(x)
            scaled = self.ns.fit_transform(scaled)
        else:
            scaled = self.sc.transform(x)
            scaled = self.ns.transform(scaled)
        return scaled
