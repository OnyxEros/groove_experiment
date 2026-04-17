from sklearn.metrics import r2_score


def evaluate(model, df):
    features = ["D", "V", "S_real", "u1", "u2", "u3"]
    X = df[features].values
    y = df["response"].values

    preds = model.predict(X)

    return {
        "r2": r2_score(y, preds)
    }
