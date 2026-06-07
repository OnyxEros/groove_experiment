"""
perception_space/core/loader.py  — PATCH
==========================================

CORRECTION : labels.npy absent → fallback gracieux
    Le run courant utilise le pipeline 'groove' qui inclut ProjectionStep
    mais pas ClusteringStep → clustering/labels.npy n'existe pas.

    Avant : FileNotFoundError → crash total du pipeline
    Après : warning + clusters synthétiques (tous à 0) pour permettre
            au reste du pipeline de tourner sans clustering.

    Un cluster unique n'empêche pas le Mantel, l'ICC ou la géométrie locale.
    Les figures cluster_groove seront triviales mais ne planteront pas.

INTÉGRATION : remplacer load_analysis_run() dans
    perception_space/core/loader.py
"""

from pathlib import Path
import warnings
import numpy as np
import json


def load_analysis_run(run_dir: Path) -> dict:
    embeddings_dir = run_dir / "embeddings"

    structural = np.load(embeddings_dir / "structural.npy")
    realized   = np.load(embeddings_dir / "realized.npy")
    summary    = json.loads((run_dir / "summary.json").read_text())

    n = realized.shape[0]

    # ── Clusters — fallback si labels.npy absent ──────────────────────────
    labels_path = run_dir / "clustering" / "labels.npy"
    if labels_path.exists():
        clusters = np.load(labels_path)
        print(f"  [loader] clusters         : {len(np.unique(clusters))} clusters chargés depuis labels.npy")
    else:
        # Pipeline 'groove' ne produit pas ClusteringStep → fallback 1 cluster
        clusters = np.zeros(n, dtype=int)
        warnings.warn(
            f"\n[loader] labels.npy absent dans {labels_path}\n"
            f"  Le pipeline courant n'inclut pas ClusteringStep.\n"
            f"  Fallback : tous les stimuli dans le cluster 0.\n"
            f"  Pour obtenir les clusters : relancer avec --analysis (mode 'full')\n"
            f"  ou ajouter 'clustering' au pipeline dans analysis/core/run.py.",
            UserWarning,
            stacklevel=2,
        )
        print(f"  [loader] ⚠  labels.npy absent — fallback cluster unique (tous C0)")

    # ── Mapping stim_id → row index ───────────────────────────────────────
    stim_id_map_path = run_dir / "stim_id_map.json"

    if stim_id_map_path.exists():
        stim_id_map    = json.loads(stim_id_map_path.read_text())
        stim_id_to_row = {sid: i for i, sid in enumerate(stim_id_map)}
    else:
        stim_id_map, stim_id_to_row = _rebuild_stim_id_map(run_dir, n)

    # ── UMAP 2D (optionnel) ───────────────────────────────────────────────
    umap_2d      = None
    umap_2d_path = embeddings_dir / "umap_2d.npy"

    if umap_2d_path.exists():
        try:
            umap_2d = np.load(umap_2d_path)
            if umap_2d.shape[0] != n:
                warnings.warn(
                    f"umap_2d.npy shape mismatch ({umap_2d.shape[0]} vs {n}) — ignoré",
                    UserWarning, stacklevel=2,
                )
                umap_2d = None
        except Exception:
            umap_2d = None

    # ── Sanity checks ─────────────────────────────────────────────────────
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
        "stim_id_map":    stim_id_map,
        "stim_id_to_row": stim_id_to_row,
        "umap_2d":        umap_2d,
    }


def _rebuild_stim_id_map(run_dir: Path, n: int) -> tuple[list[str], dict[str, int]]:
    candidates = [
        run_dir / "metadata.csv",
        run_dir.parent / "metadata.csv",
        run_dir.parent.parent / "data" / "metadata.csv",
        Path("data") / "metadata.csv",
        Path("metadata.csv"),
    ]

    meta_path = None
    for p in candidates:
        if p.exists():
            meta_path = p
            break

    if meta_path is None:
        raise RuntimeError(
            "\n╔══════════════════════════════════════════════════════════════╗\n"
            "║  ERREUR CRITIQUE — stim_id_map.json absent (P1)             ║\n"
            "╠══════════════════════════════════════════════════════════════╣\n"
            "║  Récupération :                                              ║\n"
            "║    python cli.py --new-run && python cli.py --analysis       ║\n"
            "╚══════════════════════════════════════════════════════════════╝"
        )

    try:
        import pandas as pd
        meta = pd.read_csv(meta_path)
    except Exception as e:
        raise RuntimeError(
            f"stim_id_map.json absent et metadata.csv illisible ({meta_path}) : {e}\n"
            "→ python cli.py --new-run && python cli.py --analysis"
        ) from e

    if "stim_id" in meta.columns:
        stim_ids = meta["stim_id"].astype(str).tolist()
    elif "id" in meta.columns:
        stim_ids = [f"stim_{int(i):04d}" for i in meta["id"]]
    else:
        raise RuntimeError(
            f"metadata.csv ({meta_path}) ne contient ni 'stim_id' ni 'id'.\n"
            "→ python cli.py --new-run && python cli.py --analysis"
        )

    if len(stim_ids) != n:
        raise RuntimeError(
            f"Reconstruction stim_id_map impossible :\n"
            f"  metadata.csv a {len(stim_ids)} lignes, realized.npy a {n} lignes.\n"
            "→ python cli.py --new-run && python cli.py --analysis"
        )

    stim_id_map    = stim_ids
    stim_id_to_row = {sid: i for i, sid in enumerate(stim_ids)}

    save_path = run_dir / "stim_id_map.json"
    try:
        save_path.write_text(json.dumps(stim_ids, indent=2))
        warnings.warn(
            f"[loader] stim_id_map reconstruit depuis {meta_path} → {save_path}",
            UserWarning, stacklevel=3,
        )
    except Exception:
        pass

    return stim_id_map, stim_id_to_row
