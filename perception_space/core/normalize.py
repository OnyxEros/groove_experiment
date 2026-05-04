import numpy as np

def normalize(X, eps=1e-8):
    return (X - X.mean(axis=0)) / (X.std(axis=0) + eps)
