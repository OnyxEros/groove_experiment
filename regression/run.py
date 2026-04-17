from regression.data_loader import load_dataset
from regression.model import build_X, get_target, train_ridge, train_ols
from regression.evaluation import evaluate_model


def run_regression():
    df = load_dataset()

    X, features = build_X(df)
    y = get_target(df)

    # OLS (interpretation paper)
    ols = train_ols(X, y)
    ols_metrics = evaluate_model(ols, X, y)

    # Ridge (stable)
    ridge = train_ridge(X, y, alpha=1.0)
    ridge_metrics = evaluate_model(ridge, X, y)

    print("\n📊 OLS RESULTS")
    print(ols_metrics)

    print("\n📊 RIDGE RESULTS")
    print(ridge_metrics)

    return {
        "ols": ols,
        "ridge": ridge,
        "features": features
    }