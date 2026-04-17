from sklearn.linear_model import Ridge


def train_model(df):

    features = ["D", "V", "S_real", "u1", "u2", "u3"]
    target = "response"

    df = df.dropna(subset=features + [target])

    X = df[features].values
    y = df[target].values

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    return model
