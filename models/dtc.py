from sklearn.tree import DecisionTreeClassifier

from models.imodel import IModel


class DtcModel(IModel):
    def __init__(self, **kwargs):
        _kwargs = {
            "criterion": "gini",
            "splitter": "best"
        }
        _kwargs = {**_kwargs, **kwargs}
        super().__init__(**_kwargs)
        self.name = "Decision Tree Classifier"

    def create_model(self, preprocess=False):
        self.model = DecisionTreeClassifier(**self.kwargs)
