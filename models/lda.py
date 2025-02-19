from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from models.imodel import IModel


class LdaModel(IModel):
    def __init__(self):
        kwargs = {
            "multi_class": "multinomial",
            "solver": "svd"
        }
        super().__init__(**kwargs)
        self.name = "Linear Discriminant Analysis"

    def create_model(self, preprocess=False):
        self.model = LinearDiscriminantAnalysis(solver=self.solver)
