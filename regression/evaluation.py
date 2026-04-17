from sklearn.metrics import r2_score


def evaluate_model(model, df):
    X = df[model.feature_names_in_] if hasattr(model, "feature_names_in_") else df.values
    y = df["groove_rating"]

    pred = model.predict(X)

    return {
        "r2": r2_score(y, pred)
    }