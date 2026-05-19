"""
perception_space/core/align.py
================================
Aligne les embeddings avec les ratings perceptifs via stim_id_to_row.

AVANT (bugué) : df["stimulus_id"].astype(int) → utilisé comme index de ligne
                → corrompu si le design a été permuté (ce qui est toujours le cas)

APRÈS (correct) : utilise stim_id_to_row[stim_id] → row index garanti
"""

import pandas as pd
import numpy as np

from perception_space.core.validation import validate_perception_df


def align_embeddings_with_perception(
    embeddings:     np.ndarray,
    df:             pd.DataFrame,
    stim_id_to_row: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aligne embeddings × ratings via stim_id_to_row.

    Args:
        embeddings     : np.ndarray (n_total, d)
        df             : DataFrame avec colonnes stimulus_id, groove, complexity
        stim_id_to_row : dict { stim_id_str → row_index_in_embeddings }
                         Si None, fallback sur cast int (rétro-compat, risqué)

    Returns:
        X_aligned    : (n_aligned, d)
        y_groove     : (n_aligned,)
        y_complexity : (n_aligned,)
    """
    df = df.copy()
    validate_perception_df(df)

    df["stimulus_id"] = df["stimulus_id"].astype(str)

    if stim_id_to_row is not None:
        # ── Chemin correct ────────────────────────────────────
        valid_mask = df["stimulus_id"].isin(stim_id_to_row)
        n_invalid  = (~valid_mask).sum()
        if n_invalid > 0:
            print(f"[align] Warning: {n_invalid} stim_id absents du mapping ignorés")

        df = df[valid_mask].copy()

        if df.empty:
            raise ValueError(
                "Aucun stim_id commun entre les ratings et le mapping embeddings.\n"
                f"  ratings sample  : {df['stimulus_id'].head(3).tolist()}\n"
                f"  mapping sample  : {list(stim_id_to_row.keys())[:3]}"
            )

        row_indices  = df["stimulus_id"].map(stim_id_to_row).values
        X_aligned    = embeddings[row_indices]

    else:
        # stim_id_to_row est TOUJOURS requis depuis que run_experiment()
        # permute le design (ordre d'insertion ≠ index de ligne).
        # Un cast int→index serait silencieusement faux : on lève une erreur
        # explicite pour forcer la régénération du mapping.
        raise ValueError(
            "stim_id_to_row est absent. "
            "L'alignement par cast int est incorrect quand le design est permuté "
            "(ce qui est toujours le cas depuis groove/generator.py).\n"
            "Relance l'analyse pour régénérer stim_id_map.json :\n"
            "  python cli.py --new-run && python cli.py --analysis"
        )

    y_groove     = df["groove"].values
    y_complexity = df["complexity"].values

    return X_aligned, y_groove, y_complexity