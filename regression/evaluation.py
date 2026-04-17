from sklearn.metrics import r2_score, mean_squared_error
import numpy as np


def evaluate_model(model, X, y):
    y_pred = model.predict(X)

    return {
        "r2": r2_score(y, y_pred),
        "rmse": np.sqrt(mean_squared_error(y, y_pred))
    }