from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from models.imodel import IModel


class AbModel(IModel):
    def __init__(self):
        kwargs = {
            "n_estimators": 50,
            "learning_rate": 1.0,
            "algorithm": "SAMME",
            "random_state": 42  # Again, the ultimate answer
        }
        self.__base_estimator = DecisionTreeClassifier(max_depth=1)
        super().__init__(**kwargs)
        self.name = "AdaBoost Classifier"

    def create_model(self, preprocess=False):
        self.model = AdaBoostClassifier(self.__base_estimator, **self.kwargs)

    def set_base_estimator(self, estimator):
        self.__base_estimator = estimator
