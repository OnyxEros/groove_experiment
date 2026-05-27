"""
perception_space/core/loader.py
================================
Charge un run d'analyse et retourne les embeddings alignés sur les stim_id.

IMPORTANT : realized.npy[i] correspond au stim_id stim_id_map[i].
Le mapping est chargé depuis stim_id_map.json (produit par ExportStep).
Sans ce mapping, l'alignement embeddings × ratings est corrompu.

umap_2d.npy est sauvegardé par ExportStep si ProjectionStep a tourné.
Sans lui, _project_2d() dans run.py fait un fallback PCA.

Correction P1 :
    Avant : si stim_id_map.json est absent, le fallback construit
            {stim_0000: 0, stim_0001: 1, ...} en supposant que l'ordre
            d'insertion dans realized.npy correspond à l'ordre des stim_id.
            Or run_experiment() permute TOUJOURS le design via rng.permutation —
            stim_0000 n'est donc jamais à la ligne 0 de realized.npy.
            Résultat : tous les embeddings sont silencieusement mal alignés
            sur les ratings, corrompant UMAP, ICC, et le test de Mantel.

    Après : si stim_id_map.json est absent ET que metadata.csv est disponible,
            on tente de reconstruire le mapping depuis metadata.csv (colonne id/stim_id).
            Si la reconstruction échoue, on lève une RuntimeError explicite
            avec les instructions pour régénérer le mapping.
            Le fallback silencieusement faux est supprimé.

    Procédure de récupération si stim_id_map.json est absent :
        python cli.py --new-run && python cli.py --analysis
    Cela régénère stim_id_map.json via ExportStep.
"""

from pathlib import Path
import warnings
import numpy as np
import json


def load_analysis_run(run_dir: Path) -> dict:
    """
    Charge les artefacts d'un run d'analyse.

    Returns:
        dict {
            structural:     np.ndarray (n, d_struct)
            realized:       np.ndarray (n, d_real)
            clusters:       np.ndarray (n,) labels de cluster
            summary:        dict (summary.json)
            stim_id_map:    list[str]  index → stim_id
            stim_id_to_row: dict[str, int]  stim_id → row index dans realized
            umap_2d:        np.ndarray (n, 2) ou None
        }

    Raises:
        RuntimeError : si stim_id_map.json est absent et non reconstructible.
    """
    embeddings_dir = run_dir / "embeddings"

    structural = np.load(embeddings_dir / "structural.npy")
    realized   = np.load(embeddings_dir / "realized.npy")
    clusters   = np.load(run_dir / "clustering" / "labels.npy")
    summary    = json.loads((run_dir / "summary.json").read_text())

    n = realized.shape[0]

    # ── Mapping stim_id → row index (critique pour l'alignement) ──
    stim_id_map_path = run_dir / "stim_id_map.json"

    if stim_id_map_path.exists():
        stim_id_map    = json.loads(stim_id_map_path.read_text())
        stim_id_to_row = {sid: i for i, sid in enumerate(stim_id_map)}

    else:
        # P1 : tentative de reconstruction depuis metadata.csv
        stim_id_map, stim_id_to_row = _rebuild_stim_id_map(run_dir, n)

    # ── UMAP 2D (optionnel) ────────────────────────────────
    umap_2d      = None
    umap_2d_path = embeddings_dir / "umap_2d.npy"

    if umap_2d_path.exists():
        try:
            umap_2d = np.load(umap_2d_path)
            if umap_2d.shape[0] != n:
                warnings.warn(
                    f"umap_2d.npy shape mismatch "
                    f"({umap_2d.shape[0]} vs {n}) — ignoré",
                    UserWarning,
                    stacklevel=2,
                )
                umap_2d = None
        except Exception:
            umap_2d = None

    # ── Sanity checks ──────────────────────────────────────
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


# =========================================================
# RECONSTRUCTION DU MAPPING (P1)
# =========================================================

def _rebuild_stim_id_map(run_dir: Path, n: int) -> tuple[list[str], dict[str, int]]:
    """
    Tente de reconstruire stim_id_map depuis metadata.csv.

    La reconstruction est UNIQUEMENT valide si metadata.csv est trié
    dans le même ordre que celui utilisé lors de la génération de realized.npy.
    Cet ordre correspond à l'ordre de run_experiment() APRÈS permutation,
    c'est-à-dire l'ordre des lignes dans metadata.csv tel que généré par
    build_audio_map() / ExportStep.

    Si la reconstruction réussit, un warning est émis et le mapping
    reconstruit est sauvegardé dans stim_id_map.json pour les runs futurs.

    Si la reconstruction échoue (metadata.csv absent ou nombre de lignes
    incompatible), une RuntimeError est levée avec les instructions de
    récupération.

    Raises:
        RuntimeError : si stim_id_map.json est absent et non reconstructible.
    """
    # Chercher metadata.csv dans les emplacements courants
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
            "\n"
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║  ERREUR CRITIQUE — stim_id_map.json absent (P1)             ║\n"
            "╠══════════════════════════════════════════════════════════════╣\n"
            "║  Sans ce fichier, l'alignement embeddings × ratings est     ║\n"
            "║  corrompu : run_experiment() permute toujours le design,    ║\n"
            "║  donc stim_0000 n'est PAS à la ligne 0 de realized.npy.    ║\n"
            "║                                                              ║\n"
            "║  Récupération :                                              ║\n"
            "║    python cli.py --new-run && python cli.py --analysis       ║\n"
            "║                                                              ║\n"
            "║  Cela régénère stim_id_map.json via ExportStep.             ║\n"
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

    # Résoudre la colonne stim_id
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
            f"  Le run d'analyse est probablement corrompu ou incomplet.\n"
            "→ python cli.py --new-run && python cli.py --analysis"
        )

    # Reconstruction réussie — sauvegarder pour les runs futurs
    stim_id_map    = stim_ids
    stim_id_to_row = {sid: i for i, sid in enumerate(stim_ids)}

    save_path = run_dir / "stim_id_map.json"
    try:
        save_path.write_text(json.dumps(stim_ids, indent=2))
        warnings.warn(
            f"\n[loader] stim_id_map.json absent — mapping reconstruit depuis {meta_path}\n"
            f"  Fichier sauvegardé : {save_path}\n"
            f"  ⚠️  Valide UNIQUEMENT si metadata.csv est dans l'ordre de realized.npy.\n"
            f"  Pour garantir l'alignement : python cli.py --new-run && python cli.py --analysis",
            UserWarning,
            stacklevel=3,
        )
    except Exception:
        warnings.warn(
            f"[loader] stim_id_map reconstruit depuis {meta_path} "
            f"mais impossible à sauvegarder dans {save_path}.",
            UserWarning,
            stacklevel=3,
        )

    return stim_id_map, stim_id_to_row