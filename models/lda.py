from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from models.imodel import IModel


class LdaModel(IModel):
    def __init__(self, **kwargs):
        _kwargs = {
            "multi_class": "multinomial",
            "solver": "svd"
        }
        _kwargs = {**_kwargs, **kwargs}
        super().__init__(**_kwargs)
        self.name = "Linear Discriminant Analysis"

    def create_model(self, preprocess=False):
        self.model = LinearDiscriminantAnalysis(solver=self.solver)
