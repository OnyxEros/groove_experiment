"""
Script de diagnostic rapide des descripteurs émergents.
Lance depuis la racine du projet : python inspect_descriptors.py
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from groove.generator import run_experiment

df, _ = run_experiment(seed=42)

desc_cols = ["D", "I", "V", "S", "E", "P"]
param_cols = ["S_mv", "D_mv", "E_mv", "P_mv"]

print("=" * 60)
print("STATISTIQUES DESCRIPTEURS EMERGENTS")
print("=" * 60)
stats = df[desc_cols].describe().round(4)
print(stats.to_string())

print()
print("=" * 60)
print("VARIANCE PAR DESCRIPTEUR (cv = std/mean)")
print("=" * 60)
for c in desc_cols:
    mn = df[c].mean()
    sd = df[c].std()
    cv = sd / (abs(mn) + 1e-9)
    print(f"  {c}: mean={mn:.4f}  std={sd:.4f}  cv={cv:.3f}  "
          f"min={df[c].min():.4f}  max={df[c].max():.4f}")

print()
print("=" * 60)
print("CORRELATION PARAMS → DESCRIPTEURS")
print("=" * 60)
for p in param_cols:
    for d in desc_cols:
        r = df[[p, d]].corr().iloc[0, 1]
        bar = "#" * int(abs(r) * 20)
        print(f"  {p} → {d}: r={r:+.3f}  {bar}")
    print()

print("=" * 60)
print("CORRELATION INTER-DESCRIPTEURS")
print("=" * 60)
corr = df[desc_cols].corr().round(3)
print(corr.to_string())

print()
print("=" * 60)
print("VALEURS NULLES OU CONSTANTES")
print("=" * 60)
for c in desc_cols:
    n_zero = (df[c] == 0).sum()
    n_unique = df[c].nunique()
    print(f"  {c}: {n_zero}/{len(df)} zeros  |  {n_unique} valeurs uniques")
