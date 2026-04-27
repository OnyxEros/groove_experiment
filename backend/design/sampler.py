import hashlib
import random

def seed_from_condition(condition, base_seed=0):
    key = f"{condition['S_mv']}_{condition['D_mv']}_{condition['E']}_{base_seed}"
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % 10**8


def sample_variants(condition, generator_fn, n_variants=3, base_seed=0):
    variants = []

    for i in range(n_variants):
        seed = seed_from_condition(condition, base_seed + i)
        variants.append(generator_fn(condition, seed))

    return variants
