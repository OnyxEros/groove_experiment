"""
perception/supabase_io.py
=========================
Fetch et cache local des réponses perceptives depuis Supabase.

Corrections :
    B1 — RT_MIN abaissé de 4.0s à 1.5s (défini dans config.py — source unique).
    B3 — Filtre listen_duration ajouté.
    I2 — Détection des participants "spammeurs" (warning uniquement).
"""

import pandas as pd
from pathlib import Path

from infra.supabase_client import fetch_responses
from config import RESP_FILE, RT_MIN_S, RT_MAX_S  # ← source unique, plus de duplication

# ── Seuil d'écoute minimale (B3) ─────────────────────────
LISTEN_MIN_S = 1.5

# ── Seuil de détection spammeurs (I2) ────────────────────
SPAM_MEDIAN_RT_S = 2.5


def fetch_ratings(refresh: bool = False) -> pd.DataFrame:
    """
    Charge les réponses perceptives (cache local ou Supabase).

    Returns:
        DataFrame avec colonnes :
            participant_id, stim_id, groove, complexity, rt, created_at
        Les réponses filtrées (RT ou listen_duration hors plage) sont exclues.
    """
    cache_path = Path(RESP_FILE)

    if cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path)
        df = _validate(df)
        return df

    data = fetch_responses()

    if not data:
        raise ValueError(
            "Aucune réponse trouvée dans Supabase (table 'responses' vide).\n"
            "Lance d'abord la collecte via l'interface web, "
            "ou vérifie tes variables d'environnement SUPABASE_URL / SUPABASE_KEY."
        )

    df = pd.DataFrame(data)
    df = _validate(df)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"[supabase_io] {len(df)} réponses sauvées → {cache_path}")

    return df


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie et filtre les réponses.

    Filtres appliqués dans l'ordre :
        1. Colonnes obligatoires présentes
        2. Types numériques corrects
        3. RT hors plage [RT_MIN_S, RT_MAX_S]       (B1 : seuils depuis config.py)
        4. listen_duration < LISTEN_MIN_S            (B3 : écoute insuffisante)
        5. Détection spammeurs (médiane RT par part.) (I2 : warning uniquement)
    """
    required = {"stim_id", "groove"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans les réponses : {missing}\n"
            f"Colonnes présentes : {list(df.columns)}"
        )

    df = df.copy()
    df = df.dropna(subset=["stim_id", "groove"])
    df["stim_id"] = df["stim_id"].astype(str)
    df["groove"]  = pd.to_numeric(df["groove"],  errors="coerce")

    if "complexity" in df.columns:
        df["complexity"] = pd.to_numeric(df["complexity"], errors="coerce")

    if "rt" in df.columns:
        df["rt"] = pd.to_numeric(df["rt"], errors="coerce")

    df = df.dropna(subset=["groove"])

    # ── B1 : Filtre RT (seuils depuis config.py) ──────────
    if "rt" in df.columns:
        before = len(df)
        df = df[
            df["rt"].isna() |
            df["rt"].between(RT_MIN_S, RT_MAX_S, inclusive="both")
        ].copy()
        n_rt = before - len(df)
        if n_rt > 0:
            print(
                f"[supabase_io] {n_rt} réponse(s) filtrée(s) "
                f"RT hors [{RT_MIN_S}s–{RT_MAX_S}s]"
            )

    # ── B3 : Filtre listen_duration ───────────────────────
    if "listen_duration" in df.columns:
        df["listen_duration"] = pd.to_numeric(df["listen_duration"], errors="coerce")
        before = len(df)
        df = df[
            df["listen_duration"].isna() |
            (df["listen_duration"] >= LISTEN_MIN_S)
        ].copy()
        n_listen = before - len(df)
        if n_listen > 0:
            print(
                f"[supabase_io] {n_listen} réponse(s) filtrée(s) "
                f"listen_duration < {LISTEN_MIN_S}s (écoute insuffisante)"
            )

    # ── I2 : Détection spammeurs ──────────────────────────
    if "rt" in df.columns and "participant_id" in df.columns:
        _warn_spammers(df)

    return df


def _warn_spammers(df: pd.DataFrame) -> None:
    """
    Identifie les participants dont la médiane RT est suspicieusement basse.
    Warning uniquement — aucune donnée n'est supprimée.
    """
    rt_valid = df.dropna(subset=["rt", "participant_id"])
    if rt_valid.empty:
        return

    median_rt = rt_valid.groupby("participant_id")["rt"].median()
    spammers  = median_rt[median_rt < SPAM_MEDIAN_RT_S].sort_values()

    if spammers.empty:
        return

    print(
        f"\n[supabase_io] ⚠️  {len(spammers)} participant(s) avec médiane RT "
        f"< {SPAM_MEDIAN_RT_S}s (seuil spammeur) :"
    )
    for pid, med in spammers.items():
        n_resp = (rt_valid["participant_id"] == pid).sum()
        print(f"    participant={pid}  médiane_rt={med:.2f}s  n_réponses={n_resp}")

    print(
        "    → Vérification manuelle recommandée avant exclusion.\n"
        "      Pour exclure : ajouter les IDs à EXCLUDED_PARTICIPANTS dans config.py\n"
        "      puis relancer avec --refresh.\n"
    )