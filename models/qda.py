from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from models.imodel import IModel


class QdaModel(IModel):
    def __init__(self, **kwargs):
        _kwargs = {
        }
        _kwargs = {**_kwargs, **kwargs}
        super().__init__(**_kwargs)
        self.name = "Quadratic Discriminant Analysis"

    def create_model(self, preprocess=False):
        self.model = QuadraticDiscriminantAnalysis()
