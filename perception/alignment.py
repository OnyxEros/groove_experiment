import numpy as np
from sklearn.linear_model import Ridge


def fit_alignment(Z, ratings):
    """
    Learn mapping latent space → perception.
    """

    model = Ridge(alpha=1.0)
    model.fit(Z, ratings)

    score = model.score(Z, ratings)

    return model, score


def predict_perception(model, Z):
    return model.predict(Z)
