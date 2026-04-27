from itertools import product
import random

def generate_conditions(space, n_conditions=8, seed=42):
    random.seed(seed)

    full_grid = list(product(
        space.S_levels,
        space.D_levels,
        space.E_levels
    ))

    # sélection contrôlée (subset équilibré)
    selected = random.sample(full_grid, n_conditions)

    return [
        {"S_mv": s, "D_mv": d, "E": e}
        for s, d, e in selected
    ]
