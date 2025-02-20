from sklearn.naive_bayes import GaussianNB

from models.imodel import IModel


class GnbModel(IModel):
    def __init__(self):
        kwargs = {}
        super().__init__(**kwargs)
        self.name = "Gaussian Naive Bayes"

    def create_model(self, preprocess=False):
        self.model = GaussianNB()
