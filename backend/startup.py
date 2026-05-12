from pathlib import Path
from config import ensure_data_dirs, MP3_DIR, METADATA_PATH
import pandas as pd

# Colonnes requises dans metadata.csv (nouvelle notation)
REQUIRED_COLUMNS = {"mp3_path", "S_mv", "D_mv", "E_mv", "S"}


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
                # Rétro-compat : accepte aussi S_real à la place de S
                effective_cols = set(df.columns)
                if "S_real" in effective_cols and "S" not in effective_cols:
                    effective_cols.add("S")   # sera renommé par normalize_columns
                if "E_real" in effective_cols and "E_mv" not in effective_cols:
                    # E_real → E (émergent), pas E_mv
                    pass

                missing_cols = REQUIRED_COLUMNS - effective_cols
                if missing_cols:
                    warnings.append(
                        f"Colonnes absentes dans metadata.csv : {', '.join(sorted(missing_cols))}"
                    )

                # Vérifie les valeurs nulles sur les colonnes critiques
                for col in ("mp3_path", "S"):
                    actual_col = "S_real" if col == "S" and "S_real" in df.columns else col
                    if actual_col in df.columns and df[actual_col].isnull().any():
                        n = df[actual_col].isnull().sum()
                        warnings.append(f"{n} valeur(s) nulle(s) dans la colonne '{actual_col}'")

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