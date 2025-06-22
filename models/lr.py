from models.imodel import IModel
from sklearn.linear_model import LogisticRegression


class LrModel(IModel):
    def __init__(self, **kwargs):
        _kwargs = {
            "multi_class": "deprecated",
            "solver": "lbfgs"
        }
        _kwargs = {**_kwargs, **kwargs}
        super().__init__(**_kwargs)
        self.name = "Logistic Regression"

    def create_model(self, preprocess=False):
        self.model = LogisticRegression(multi_class=self.multi_class, solver=self.solver, verbose=False)
