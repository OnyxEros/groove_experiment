from sklearn.linear_model import Ridge
from regression.features import FEATURES, TARGET


def train_model(df):
    X = df[FEATURES].values
    y = df[TARGET].values

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    return model