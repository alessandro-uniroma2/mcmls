from sklearn.ensemble import AdaBoostClassifier

from models.imodel import IModel


class AdaBoostModel(IModel):
    def __init__(self):
        kwargs = {
            "n_estimators": 50,
            "learning_rate": 1.0
        }
        super().__init__(**kwargs)
        self.name = "AdaBoost Classifier"

    def create_model(self, preprocess=False):
        self.model = AdaBoostClassifier(**self.kwargs)
