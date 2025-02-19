from sklearn.svm import SVC

from models.imodel import IModel


class SvcModel(IModel):
    def __init__(self):
        kwargs = {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale"
        }
        super().__init__(**kwargs)
        self.name = "Support Vector Machine Classifier"

    def create_model(self, preprocess=False):
        self.model = SVC(**self.kwargs)
