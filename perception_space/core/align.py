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
        # ── Fallback rétro-compatible (risqué si design permuté) ──
        import warnings
        warnings.warn(
            "align: stim_id_to_row absent — fallback sur cast int. "
            "L'alignement peut être incorrect.",
            UserWarning,
            stacklevel=2,
        )
        try:
            idx = df["stimulus_id"].astype(int).values
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"stimulus_id non castable en int et stim_id_to_row absent : {e}"
            )

        n_embeddings = embeddings.shape[0]
        valid_mask   = (idx >= 0) & (idx < n_embeddings)
        n_invalid    = (~valid_mask).sum()
        if n_invalid > 0:
            print(f"[align] Warning: {n_invalid} stimulus_id hors plage ignorés")

        df  = df[valid_mask].copy()
        idx = idx[valid_mask]

        if df.empty:
            raise ValueError("Aucun stimulus_id valide après filtrage")

        X_aligned = embeddings[idx]

    y_groove     = df["groove"].values
    y_complexity = df["complexity"].values

    return X_aligned, y_groove, y_complexity