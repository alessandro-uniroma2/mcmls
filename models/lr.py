from models.imodel import IModel
from sklearn.linear_model import LogisticRegression


class LrModel(IModel):
    def __init__(self):
        kwargs = {
            "multi_class": "multinomial",
            "solver": "lbfgs"
        }
        super().__init__(**kwargs)
        self.name = "Logistic Regression"

    def create_model(self, preprocess=False):
        self.model = LogisticRegression(multi_class=self.multi_class, solver=self.solver)
