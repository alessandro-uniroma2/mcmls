from sklearn.tree import DecisionTreeClassifier

from models.imodel import IModel


class DtcModel(IModel):
    def __init__(self):
        kwargs = {
            "criterion": "gini",
            "splitter": "best"
        }
        super().__init__(**kwargs)
        self.name = "Decision Tree Classifier"

    def create_model(self, preprocess=False):
        self.model = DecisionTreeClassifier(**self.kwargs)
