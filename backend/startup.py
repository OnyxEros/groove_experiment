from pathlib import Path
from config import ensure_data_dirs, MP3_DIR, METADATA_PATH
import pandas as pd


REQUIRED_COLUMNS = {"mp3_path", "S_mv", "D_mv", "E", "S_real"}


def check_environment() -> dict:
    """
    Vérifie l'environnement au démarrage.
    Lève RuntimeError si une condition bloquante est détectée.
    Retourne un dict de diagnostics.
    """
    ensure_data_dirs()

    errors: list[str] = []
    warnings: list[str] = []

    # ── MP3 directory ────────────────────────────────────────
    if not MP3_DIR.exists():
        errors.append(f"Répertoire MP3 introuvable : {MP3_DIR}")
    else:
        mp3_files = list(MP3_DIR.rglob("*.mp3"))
        if not mp3_files:
            errors.append(f"Aucun fichier .mp3 dans {MP3_DIR}")
        else:
            # Vérifie les fichiers corrompus (taille 0)
            empty = [f for f in mp3_files if f.stat().st_size == 0]
            if empty:
                warnings.append(f"{len(empty)} fichier(s) MP3 vide(s) détecté(s)")

    # ── metadata.csv ─────────────────────────────────────────
    if not METADATA_PATH.exists():
        errors.append(f"metadata.csv introuvable : {METADATA_PATH}")
    else:
        try:
            df = pd.read_csv(METADATA_PATH)
        except Exception as e:
            errors.append(f"Impossible de lire metadata.csv : {e}")
            df = None

        if df is not None:
            if df.empty:
                errors.append("metadata.csv est vide")
            else:
                missing_cols = REQUIRED_COLUMNS - set(df.columns)
                if missing_cols:
                    warnings.append(
                        f"Colonnes absentes dans metadata.csv : {', '.join(sorted(missing_cols))}"
                    )

                # Vérifie les valeurs nulles sur les colonnes critiques
                for col in ("mp3_path", "S_real"):
                    if col in df.columns and df[col].isnull().any():
                        n = df[col].isnull().sum()
                        warnings.append(f"{n} valeur(s) nulle(s) dans la colonne '{col}'")

    # ── Résultat ─────────────────────────────────────────────
    if errors:
        msg = "Erreurs au démarrage :\n" + "\n".join(f"  • {e}" for e in errors)
        raise RuntimeError(msg)

    n_mp3 = len(list(MP3_DIR.rglob("*.mp3"))) if MP3_DIR.exists() else 0
    print("🎧 Environnement prêt")
    print(f"   Fichiers MP3 : {n_mp3}")
    if warnings:
        for w in warnings:
            print(f"   ⚠ {w}")

    return {"mp3_count": n_mp3, "warnings": warnings}