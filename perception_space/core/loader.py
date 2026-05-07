"""
perception_space/core/loader.py
================================
Charge un run d'analyse et retourne les embeddings alignés sur les stim_id.

IMPORTANT : realized.npy[i] correspond au stim_id stim_id_map[i].
Le mapping est chargé depuis stim_id_map.json (produit par ExportStep).
Sans ce mapping, l'alignement embeddings × ratings est corrompu.

umap_2d.npy est sauvegardé par ExportStep si ProjectionStep a tourné.
Sans lui, _project_2d() dans run.py fait un fallback PCA.
"""

from pathlib import Path
import numpy as np
import json


def load_analysis_run(run_dir: Path) -> dict:
    embeddings_dir = run_dir / "embeddings"

    structural = np.load(embeddings_dir / "structural.npy")
    realized   = np.load(embeddings_dir / "realized.npy")
    clusters   = np.load(run_dir / "clustering" / "labels.npy")
    summary    = json.loads((run_dir / "summary.json").read_text())

    # ── Mapping stim_id → row index ──────────────────────────
    stim_id_map_path = run_dir / "stim_id_map.json"
    if stim_id_map_path.exists():
        stim_id_map = json.loads(stim_id_map_path.read_text())
        stim_id_to_row = {sid: i for i, sid in enumerate(stim_id_map)}
    else:
        import warnings
        warnings.warn(
            "stim_id_map.json absent — fallback sur indexation directe. "
            "Relance l'analyse pour générer le mapping correct.",
            UserWarning, stacklevel=2,
        )
        n = realized.shape[0]
        stim_id_map    = [f"stim_{i:04d}" for i in range(n)]
        stim_id_to_row = {sid: i for i, sid in enumerate(stim_id_map)}

    # ── UMAP 2D (optionnel — produit par ExportStep depuis v2) ─
    umap_2d = None
    umap_2d_path = embeddings_dir / "umap_2d.npy"
    if umap_2d_path.exists():
        try:
            umap_2d = np.load(umap_2d_path)
            if umap_2d.shape[0] != realized.shape[0]:
                import warnings
                warnings.warn(
                    f"umap_2d.npy shape mismatch "
                    f"({umap_2d.shape[0]} vs {realized.shape[0]}) — ignoré",
                    UserWarning, stacklevel=2,
                )
                umap_2d = None
        except Exception:
            umap_2d = None

    # ── Sanity checks ─────────────────────────────────────────
    n = realized.shape[0]
    assert structural.shape[0] == n, \
        f"structural ({structural.shape[0]}) ≠ realized ({n})"
    assert clusters.shape[0] == n, \
        f"clusters ({clusters.shape[0]}) ≠ realized ({n})"
    assert len(stim_id_map) == n, \
        f"stim_id_map ({len(stim_id_map)}) ≠ realized ({n})"

    return {
        "structural":     structural,
        "realized":       realized,
        "clusters":       clusters,
        "summary":        summary,
        "stim_id_map":    stim_id_map,       # list[str], index → stim_id
        "stim_id_to_row": stim_id_to_row,    # dict str → int
        "umap_2d":        umap_2d,           # np.ndarray (n,2) ou None
    }