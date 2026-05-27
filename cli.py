"""
cli.py
======
Point d'entrée unique du système Groove Experiment.

Notation :
    Paramètres génératifs : S_mv, D_mv, E_mv, P_mv
    Descripteurs émergents : D, I, V, S, E, P

Pipeline du mémoire
-------------------
    §4.1 Générateur rythmique
        make generate    → génère les stimuli, MIDI, MP3, metadata.csv
        make validate    → figures de validation structurelle (§4.1.4)
        make preview     → écoute 3 stimuli de référence

    §4.2 Collecte des données
        make serve       → démarre le serveur de l'expérience
        make sync        → rapatrie les réponses Supabase → CSV local

    §5 Espace latent
        make new-run     → crée un dossier de run horodaté
        make analysis    → embeddings → UMAP → clustering → export

    §6 Modélisation statistique
        make regression  → Ridge · ElasticNet · SVR · RF · LMM (3 feature sets)
        make perception  → alignement Ridge espace latent → groove_mean
        make perc-space  → ICC · test de Mantel · géométrie locale

    Pipeline complet
        make thesis      → sync + analysis + regression + perception + perc-space
        make figures     → collecte toutes les figures dans figures/

Architecture des données
------------------------
    fetch_ratings()           → FORMAT LONG   (1 ligne / réponse, participant_id)
                                → perception-space (ICC, cohérence, variabilité)

    load_ratings_df()         → FORMAT AGRÉGÉ (1 ligne / stimulus)
                                → regression, perception (Ridge)

    load_perceptual_dataset() → FORMAT JOINT  (stimuli × ratings agrégés)
                                → alignment Ridge avec features acoustiques
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.panel import Panel
    from rich.columns import Columns
    from rich import box as rich_box
    _RICH = True
except ImportError:
    _RICH = False

from config import (
    ANALYSIS_DIR, METADATA_PATH, MIDI_DIR, MP3_DIR,
    PREVIEW_DIR, RESP_FILE, SOUNDFONT_PATH, WAV_DIR,
    ensure_data_dirs,
)

_console = Console() if _RICH else None


# =========================================================
# AFFICHAGE
# =========================================================

def _print(msg: str, style: str = "bold green") -> None:
    if _RICH:
        _console.print(f"\n{msg}\n", style=style)
    else:
        print(f"\n{msg}\n")


def _warn(msg: str) -> None:
    if _RICH:
        _console.print(f"⚠️  {msg}", style="yellow")
    else:
        print(f"⚠️  {msg}", file=sys.stderr)


def _error(msg: str) -> None:
    if _RICH:
        _console.print(f"\n❌  {msg}\n", style="bold red")
    else:
        print(f"\n❌  {msg}\n", file=sys.stderr)


def _info(msg: str) -> None:
    if _RICH:
        _console.print(f"   {msg}", style="dim")
    else:
        print(f"   {msg}")


def _done() -> None:
    if _RICH:
        _console.print("\n[bold green]🔥  DONE[/bold green]\n")
    else:
        print("\n🔥  DONE\n")


def safe_exit(msg: str, code: int = 1) -> None:
    _error(msg)
    sys.exit(code)


@contextmanager
def step(label: str, dry_run: bool = False):
    if dry_run:
        if _RICH:
            _console.print(f"  [dim]DRY-RUN[/dim] [cyan]{label}[/cyan]")
        else:
            print(f"  [DRY-RUN] {label}")
        yield
        return

    start = time.perf_counter()

    if _RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold cyan]{label}[/bold cyan]"),
            TimeElapsedColumn(),
            console=_console,
            transient=True,
        ) as progress:
            progress.add_task("", total=None)
            try:
                yield
            except Exception:
                elapsed = time.perf_counter() - start
                _console.print(f"  ❌ [red]{label}[/red] failed after {elapsed:.1f}s")
                _console.print_exception(show_locals=False)
                raise
    else:
        print(f"\n▶  {label}…")
        try:
            yield
        except Exception as exc:
            elapsed = time.perf_counter() - start
            print(f"  FAILED after {elapsed:.1f}s: {exc}")
            traceback.print_exc()
            raise

    elapsed = time.perf_counter() - start
    if _RICH:
        _console.print(f"  ✔  [green]{label}[/green] [dim]({elapsed:.1f}s)[/dim]")
    else:
        print(f"  ✔  {label} ({elapsed:.1f}s)")


# =========================================================
# PREFLIGHT
# =========================================================

def _check_soundfont() -> bool:
    ok = Path(SOUNDFONT_PATH).exists()
    if not ok:
        _warn(f"SoundFont not found: {SOUNDFONT_PATH}")
    return ok


def _check_deps() -> dict[str, bool]:
    result: dict[str, bool] = {}
    for pkg in ["numpy", "pandas", "scipy", "sklearn", "umap", "shap", "statsmodels"]:
        try:
            __import__(pkg.replace("-", "_"))
            result[pkg] = True
        except ImportError:
            result[pkg] = False
    return result


# =========================================================
# STATUS
# =========================================================

def cmd_status() -> None:
    if not _RICH:
        print("Status requires `rich`.  Install with: pip install rich")
        return

    table = Table(
        title="🎧  Groove Experiment — État du système",
        show_header=True,
        header_style="bold magenta",
        box=rich_box.SIMPLE_HEAVY,
    )
    table.add_column("Composant",  style="cyan",  min_width=18)
    table.add_column("Chemin",     style="dim",   min_width=36)
    table.add_column("État",       justify="center")

    dirs = {
        "MIDI dir":     MIDI_DIR,
        "MP3 dir":      MP3_DIR,
        "Preview dir":  PREVIEW_DIR,
        "Analysis dir": ANALYSIS_DIR,
        "Metadata":     METADATA_PATH,
        "SoundFont":    Path(SOUNDFONT_PATH),
    }
    for name, path in dirs.items():
        exists = path.exists()
        badge  = "[green]✔[/green]" if exists else "[red]✗[/red]"
        extra  = ""
        if exists and path.is_dir():
            n     = sum(1 for _ in path.rglob("*") if _.is_file())
            extra = f" ({n} fichiers)"
        elif exists and path.is_file():
            size  = path.stat().st_size
            extra = f" ({size/1024:.0f} KB)"
        table.add_row(name, str(path) + extra, badge)

    # Réponses locales
    resp = Path(RESP_FILE)
    if resp.exists():
        try:
            import pandas as pd
            df = pd.read_csv(resp)
            n_resp  = len(df)
            n_parts = df["participant_id"].nunique() if "participant_id" in df.columns else "?"
            n_stims = df["stim_id"].nunique() if "stim_id" in df.columns else "?"
            table.add_row(
                "responses.csv",
                str(resp),
                f"[green]✔  ({n_resp} réponses · {n_parts} participants · {n_stims} stimuli)[/green]",
            )
        except Exception:
            table.add_row("responses.csv", str(resp), "[yellow]⚠  illisible[/yellow]")
    else:
        table.add_row("responses.csv", str(resp), "[dim]–  (pas encore synchronisé)[/dim]")

    # Run courant
    current_run_file = Path(".current_run")
    if current_run_file.exists():
        run_path = Path(current_run_file.read_text().strip())
        if run_path.exists():
            figs = list(run_path.rglob("*.png")) + list(run_path.rglob("*.pdf"))
            table.add_row("Run courant", str(run_path.name), f"[green]✔  ({len(figs)} figures)[/green]")
        else:
            table.add_row("Run courant", str(run_path), "[red]✗  introuvable[/red]")

    table.add_section()
    for pkg, ok in _check_deps().items():
        badge = "[green]✔[/green]" if ok else "[yellow]–  (optionnel)[/yellow]"
        table.add_row(pkg, f"import {pkg}", badge)

    _console.print(table)


# =========================================================
# CLEAN
# =========================================================

def cmd_clean(targets: list[str], dry_run: bool = False) -> None:
    if not targets:
        targets = ["all"]

    _print(
        f"🧹  Nettoyage : {targets}" + (" [DRY-RUN]" if dry_run else ""),
        style="yellow",
    )

    dispatch = {
        "outputs":   _clean_outputs,
        "metadata":  _clean_metadata,
        "responses": _clean_responses,
        "analysis":  _clean_analysis,
        "cache":     _clean_pycache,
    }

    if "all" in targets:
        targets = list(dispatch.keys())

    for target in targets:
        if target not in dispatch:
            _warn(f"Cible inconnue : {target!r} — ignorée")
            continue
        if dry_run:
            _info(f"[DRY-RUN] supprimerait : {target}")
        else:
            dispatch[target]()

    _print("✔  Nettoyage terminé")


def _clean_outputs() -> None:
    for d in [MIDI_DIR, WAV_DIR, MP3_DIR, PREVIEW_DIR]:
        if d.exists():
            shutil.rmtree(d)
            _info(f"supprimé {d}")


def _clean_metadata() -> None:
    if METADATA_PATH.exists():
        METADATA_PATH.unlink()
        _info(f"supprimé {METADATA_PATH}")


def _clean_responses() -> None:
    resp = Path(RESP_FILE)
    if resp.exists():
        resp.unlink()
        _info(f"supprimé {resp}")
    else:
        _warn("Pas de cache de réponses local à supprimer.")


def _clean_analysis(subdirs: list[str] | None = None) -> None:
    if not ANALYSIS_DIR.exists():
        return
    if subdirs is None:
        shutil.rmtree(ANALYSIS_DIR)
        _info(f"supprimé {ANALYSIS_DIR}")
        from config import _CURRENT_RUN_FILE
        if _CURRENT_RUN_FILE.exists():
            _CURRENT_RUN_FILE.unlink()
            _info("supprimé .current_run")
    else:
        for sub in subdirs:
            p = ANALYSIS_DIR / sub
            if p.exists():
                shutil.rmtree(p)


def _clean_pycache() -> None:
    removed = 0
    for d in Path(".").rglob("__pycache__"):
        try:
            shutil.rmtree(d)
            removed += 1
        except Exception as exc:
            _warn(f"Impossible de supprimer {d}: {exc}")
    for f in Path(".").rglob("*.pyc"):
        try:
            f.unlink()
            removed += 1
        except Exception:
            pass
    _info(f"{removed} entrées cache supprimées")


# =========================================================
# §4.1 — GÉNÉRATION
# =========================================================

def cmd_generate(
    seed: int,
    n_repeats: int | None,
    skip_audio: bool = False,
    dry_run: bool = False,
    force: bool = False, 
) -> None:
    """
    §4.1  Génère les stimuli, exporte les MIDI, rend les MP3.

    Produit :
        data/midi/          — fichiers MIDI par stimulus
        data/mp3/           — fichiers MP3 normalisés (EBU R128 –16 LUFS)
        data/metadata.csv   — design × descripteurs émergents × chemins audio
    """
    # ── Verrou campagne ───────────────────────────────────────────────────────
    lock_file = BASE_DIR / ".groove_locked"
    if lock_file.exists() and not force:
        msg = lock_file.read_text().strip()
        safe_exit(
            f"Génération bloquée — campagne en cours.\n"
            f"  {lock_file} : {msg}\n\n"
            f"  Pour forcer (DANGER) : python cli.py --generate --force"
        )
    # ── fin verrou ────────────────────────────────────────────────────────────
    
    if dry_run:
        _print("🎛️   [DRY-RUN] Pipeline de génération — rien ne sera écrit")
        return

    from groove.generator import run_experiment
    from audio.midi_export import export_all
    from audio.mp3 import convert_all, build_audio_map

    ensure_data_dirs()

    with step("Générer les stimuli  [groove.generator]"):
        df, stim_cache = run_experiment(seed=seed, n_repeats=n_repeats)
        _info(f"{len(df)} stimuli générés  (seed={seed})")

    with step("Exporter en MIDI  [audio.midi_export]"):
        export_all(df, stim_cache, out_dir=MIDI_DIR)

    if skip_audio:
        _warn("Rendu audio ignoré (--skip-audio)")
    else:
        if not _check_soundfont():
            safe_exit(
                f"SoundFont introuvable : {SOUNDFONT_PATH}\n"
                "Utilisez --skip-audio pour contourner."
            )
        with step("Rendre l'audio  WAV → MP3 (EQ + EBU R128)"):
            convert_all(
                midi_root=MIDI_DIR,
                wav_root=WAV_DIR,
                mp3_root=MP3_DIR,
                soundfont=str(SOUNDFONT_PATH),
            )

    with step("Construire metadata.csv  [build_audio_map]"):
        df = build_audio_map(df, mp3_root=MP3_DIR)
        METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(METADATA_PATH, index=False)

    n_mp3 = len(list(MP3_DIR.rglob("*.mp3"))) if MP3_DIR.exists() else 0
    _print(
        f"✔  Génération terminée — {len(df)} stimuli · {n_mp3} MP3 → {METADATA_PATH}"
    )


# =========================================================
# §4.1.4 — VALIDATION STRUCTURELLE DU GÉNÉRATEUR
# =========================================================

def cmd_validate(dry_run: bool = False) -> None:
    """
    §4.1.4  Génère les figures de validation structurelle du générateur.

    Produit (dans le run courant / figures/) :
        generative_validation.pdf  — matrice de couplage, stabilité stochastique,
                                     distribution topologique, VIF (Panel A–D)
        dataset_structure.pdf      — corrélations inter-descripteurs, ACP,
                                     violins de sensibilité, UMAP (Panel A–D)

    Ces figures correspondent directement à la section §4.1.4 du mémoire :
    « Validation du moteur de synthèse rythmique ».
    """
    if dry_run:
        _print("🔬  [DRY-RUN] Validation structurelle ignorée")
        return

    from analysis.viz.generative_validation import GenerativeValidation
    from analysis.viz.dataset_structure import DatasetStructureFigure
    from analysis.dataset.loader import load_dataset
    from config import get_current_run

    with step("Charger le corpus  [metadata.csv]"):
        df = load_dataset()
        _info(f"{len(df)} stimuli chargés")

    run_dir = get_current_run()
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    with step("Validation générative  (Panel A–D)"):
        GenerativeValidation().plot(
            df=df,
            path=fig_dir / "generative_validation.pdf",
            verbose=True,
        )

    with step("Structure du dataset  (Panel A–D)"):
        DatasetStructureFigure().plot(
            df=df,
            path=fig_dir / "dataset_structure.pdf",
            verbose=True,
        )

    _print(f"✔  Validation terminée → {fig_dir}")


# =========================================================
# §4.1  — PREVIEW
# =========================================================

def cmd_preview(seed: int = 42, dry_run: bool = False) -> None:
    """
    Génère 3 stimuli de référence pour écoute comparative :
        baseline    S_mv=0 D_mv=1 E_mv=0.0 P_mv=0  — groove mécanique
        swing       S_mv=0 D_mv=1 E_mv=0.5 P_mv=0  — swing modéré
        syncopated  S_mv=2 D_mv=1 E_mv=0.0 P_mv=0  — syncopation forte
    """
    if dry_run:
        _print("🎧  [DRY-RUN] Preview ignoré")
        return

    import numpy as np
    import pandas as pd
    from groove.generator import Grid, MicroTiming, Stimulus, Voices, _derive_seed
    from audio.midi_export import export_all
    from audio.mp3 import convert_all

    configs = [
        {"name": "baseline",   "phase": 0, "S_mv": 0, "D_mv": 1, "E_mv": 0.0, "P_mv": 0},
        {"name": "swing",      "phase": 0, "S_mv": 0, "D_mv": 1, "E_mv": 0.5, "P_mv": 0},
        {"name": "syncopated", "phase": 0, "S_mv": 2, "D_mv": 1, "E_mv": 0.0, "P_mv": 0},
    ]

    with step("Construire les stimuli de preview"):
        grid    = Grid()
        voices  = Voices(grid, seed=seed)
        rows, cache = [], {}
        for i, cfg in enumerate(configs):
            hihat_seed  = _derive_seed(seed, i, 0)
            timing_seed = _derive_seed(seed, i, 1)
            micro   = MicroTiming(
                rng=np.random.default_rng(timing_seed),
                step_duration=grid.step_duration,
            )
            builder = Stimulus(voices, micro)
            stim = builder.build(cfg, seed=hihat_seed)
            cache[i] = stim
            rows.append({"id": i, "stim_id": f"preview_{cfg['name']}", **cfg})

        df = pd.DataFrame(rows)
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    with step("Exporter MIDI preview"):
        export_all(df, cache, out_dir=PREVIEW_DIR)

    with step("Rendre audio preview"):
        convert_all(
            midi_root=PREVIEW_DIR,
            wav_root=PREVIEW_DIR / "wav",
            mp3_root=PREVIEW_DIR / "mp3",
            soundfont=str(SOUNDFONT_PATH),
        )

    _print(f"✔  Preview prêt → {PREVIEW_DIR / 'mp3'}")
    for cfg in configs:
        _info(f"  {cfg['name']:<12} S_mv={cfg['S_mv']} D_mv={cfg['D_mv']} E_mv={cfg['E_mv']}")


# =========================================================
# §4.2 — SERVEUR
# =========================================================

def cmd_serve(dry_run: bool = False) -> None:
    """
    §4.2  Démarre le serveur FastAPI de l'expérience perceptive.

    Accessible sur http://localhost:8000
    Participants évaluent 30 stimuli (groove + complexité sur échelle 1–7).
    """
    if dry_run:
        _print("🌐  [DRY-RUN] Serveur ignoré")
        return

    import uvicorn
    from config import PORT

    _print(f"🌐  Démarrage du serveur → http://localhost:{PORT}")
    uvicorn.run(
        "backend.api:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
        log_level="info",
    )


# =========================================================
# §4.2 — SYNC SUPABASE
# =========================================================

def cmd_sync(dry_run: bool = False) -> None:
    """
    §4.2  Rapatrie les réponses depuis Supabase vers le cache local.

    Produit :
        data/responses.csv  — format long (1 ligne / réponse)
                              colonnes : participant_id, stim_id, groove,
                              complexity, rt, listen_duration, musical_background
    """
    if dry_run:
        _print("☁️   [DRY-RUN] Sync Supabase ignoré")
        return

    import pandas as pd
    from infra.supabase_client import fetch_responses

    with step("Rapatrier les réponses ← Supabase"):
        data = fetch_responses()
        if not data:
            safe_exit("Aucune réponse dans Supabase (table 'responses' vide).")
        df = pd.DataFrame(data)

    with step("Écrire le cache local"):
        cache_path = Path(RESP_FILE)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)

    n_parts = df["participant_id"].nunique() if "participant_id" in df.columns else "?"
    n_stims = df["stim_id"].nunique() if "stim_id" in df.columns else "?"
    _print(
        f"✔  {len(df)} réponses  ·  {n_parts} participants"
        f"  ·  {n_stims} stimuli  →  {RESP_FILE}"
    )


# =========================================================
# §5 — RUN + ANALYSIS
# =========================================================

def cmd_new_run() -> None:
    """§5  Crée un dossier de run horodaté dans data/analysis/."""
    from config import new_run
    run_dir = new_run()
    _print(f"✔  Run courant : {run_dir}")


def cmd_analysis(
    mode: str = "groove",
    steps: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    """
    §5  Pipeline d'analyse de l'espace latent.

    Modes disponibles :
        groove  (défaut) — embeddings → clustering → interpretation → viz → export
        full             — embeddings → projection UMAP → viz → export
        full_with_clustering — full + clustering + metrics

    Produit dans le run courant :
        embeddings/structural.npy   — espace génératif (S_mv, D_mv, E_mv, P_mv)
        embeddings/realized.npy     — espace perceptif (D, S, E, P)
        embeddings/umap_2d.npy      — projection 2D
        clustering/labels.npy       — labels de cluster k-means
        figures/                    — validation générative + structure dataset
        stim_id_map.json            — mapping stim_id → ligne dans realized.npy
        summary.json                — snapshot de debug
    """
    if dry_run:
        _print(f"🧠  [DRY-RUN] Analysis — mode={mode}, steps={steps or 'default'}")
        return

    from analysis.core.run import run_analysis as _engine

    with step(f"Pipeline d'analyse  [mode={mode}]"):
        _engine(mode=mode, steps=steps, save=True, seed=42)


# =========================================================
# §6 — RÉGRESSION
# =========================================================

def cmd_regression(
    feature_set:    str  = "all",
    refresh:        bool = False,
    check_db:       bool = True,
    exclude_single: bool = True,
    dry_run:        bool = False,
) -> None:
    """
    §6.2  Régression groove_mean ~ features acoustiques/génératifs.

    Modèles : Ridge · ElasticNet · SVR · RandomForest · LMM (REML)

    Feature sets :
        design    — paramètres manipulés (S_mv, D_mv, E_mv, P_mv)
        acoustic  — descripteurs réalisés (D, I, V, S, E, P)
        all       — union design + acoustic
        interactions — espace orthogonalisé + termes croisés (D², D×P, S×E, D×S)

    Paramètres :
        exclude_single (défaut True) — exclut les n=1 (groove_std artificiel)
        --include-single             — reproduit le run v1 avec ces stimuli
    """
    if dry_run:
        single_tag = "exclu" if exclude_single else "inclus"
        _print(
            f"📈  [DRY-RUN] Régression — features={feature_set}, n=1 {single_tag}"
        )
        return

    from regression.run import run_regression

    single_tag = "exclu" if exclude_single else "inclus"
    with step(f"Régression  [features={feature_set} · n=1 {single_tag}]"):
        result = run_regression(
            feature_set=feature_set,
            refresh=refresh,
            check_db=check_db,
            exclude_single=exclude_single,
        )

    r2   = result.get("best_r2")
    best = result.get("best_model", "?")
    _print(
        f"✔  Régression terminée — meilleur={best}"
        + (f"  R²={r2:.3f}" if r2 is not None else "")
    )


def cmd_regression_all(
    refresh:        bool = False,
    check_db:       bool = True,
    exclude_single: bool = True,
    dry_run:        bool = False,
) -> None:
    """§6.2  Lance la régression sur les 3 feature sets (design / acoustic / all)."""
    for fs in ("design", "acoustic", "all"):
        cmd_regression(
            feature_set=fs,
            refresh=(refresh and fs == "design"),
            check_db=(check_db and fs == "design"),
            exclude_single=exclude_single,
            dry_run=dry_run,
        )


# =========================================================
# §6.3 — ALIGNEMENT PERCEPTIF
# =========================================================

def cmd_perception(refresh: bool = False, dry_run: bool = False) -> None:
    """
    §6.3  Alignement Ridge : espace latent acoustique → groove_mean.

    Source : load_perceptual_dataset() — format joint, 1 ligne / stimulus.
    Features : descripteurs émergents D, I, V, S, E (réalisés).
    Cible    : groove_mean (moyenne inter-participants par stimulus).
    """
    if dry_run:
        _print("🧠  [DRY-RUN] Alignement perceptif ignoré")
        return

    import pandas as pd
    from perception.alignment import fit_alignment, print_alignment_report
    from perception.loader import load_perceptual_dataset
    from perception.metrics import cluster_perception_diff

    with step("Charger le dataset joint  [agrégé par stimulus]"):
        meta = pd.read_csv(METADATA_PATH)
        if "stim_id" not in meta.columns and "id" in meta.columns:
            meta = meta.rename(columns={"id": "stim_id"})

        df = load_perceptual_dataset(embedding_df=meta, refresh=refresh)
        feature_cols = [c for c in ["D", "I", "V", "S", "E"] if c in df.columns]

        if not feature_cols:
            safe_exit(
                "Aucun descripteur émergent dans le dataset joint.\n"
                "Colonnes disponibles : " + str(list(df.columns)) + "\n"
                "Vérifiez que metadata.csv contient D, I, V, S, E."
            )

        if "groove_mean" not in df.columns:
            safe_exit(
                "groove_mean absent du dataset joint.\n"
                "Lancez d'abord : make sync"
            )

    with step(f"Alignement Ridge  [features={feature_cols}]"):
        model, metrics = fit_alignment(
            df[feature_cols].values,
            df["groove_mean"].values,
        )
        print_alignment_report(metrics, label="+".join(feature_cols))

    r2 = metrics.get("r2_cv_mean", metrics.get("r2"))
    _print(f"📊  Alignement perceptif R² CV = {r2:.4f}")

    if "cluster" in df.columns:
        scores = cluster_perception_diff(
            df["cluster"].values,
            df["groove_mean"].values,
        )
        _print(f"📦  Groove moyen par cluster : {scores}")


# =========================================================
# §6.4 — ESPACE PERCEPTIF
# =========================================================

def cmd_perception_space(refresh: bool = False, dry_run: bool = False) -> None:
    """
    §6.4  Géométrie de l'espace perceptif.

    Calcule :
        ICC(2,1) inter-participants  — fiabilité des jugements
        Test de Mantel               — corrélation distance latente / écart groove
        Géométrie locale k-NN        — mean, std, slope, agreement par stimulus
        Variabilité inter-stimuli    — stimuli ambigus vs consensus

    Source : fetch_ratings() — FORMAT LONG (1 ligne / réponse, participant_id requis).

    Produit dans le run courant / perception_space/ :
        figures/umap_groove.png
        figures/cluster_groove.png
        figures/local_geometry_groove.png
        figures/permutation_test.png
        figures/icc_summary.png
        figures/stimulus_variance.png
        figures/effect_*.png         (si musical_background disponible)
    """
    if dry_run:
        _print("🧠  [DRY-RUN] Espace perceptif ignoré")
        return

    from perception_space.run import run_perception_space
    from perception.supabase_io import fetch_ratings

    with step("Charger les ratings  [format long, participant_id présent]"):
        df = fetch_ratings(refresh=refresh)
        df = df.rename(columns={"stim_id": "stimulus_id"})
        df["stimulus_id"] = df["stimulus_id"].astype(str)
        df = df.dropna(subset=["groove"])

        if "complexity" in df.columns and df["complexity"].isna().any():
            median_c = df["complexity"].median()
            df["complexity"] = df["complexity"].fillna(median_c)

        n_resp  = len(df)
        n_parts = df["participant_id"].nunique() if "participant_id" in df.columns else "?"
        n_stims = df["stimulus_id"].nunique()
        _info(
            f"{n_resp} réponses  ·  {n_parts} participants"
            f"  ·  {n_stims} stimuli"
        )

    with step("Géométrie de l'espace perceptif  [ICC · Mantel · k-NN]"):
        results = run_perception_space(perception_data=df)

    icc   = results.get("icc", float("nan"))
    interp = results.get("icc_interp", "?")
    mantel_r = results.get("mantel_r", float("nan"))
    mantel_p = results.get("mantel_p", float("nan"))

    _print(
        f"✔  ICC = {icc:.3f} ({interp})  ·  "
        f"Mantel r = {mantel_r:.3f}  p = {mantel_p:.4f}"
    )


# =========================================================
# COLLECTE DES FIGURES DU MÉMOIRE
# =========================================================

def cmd_figures(out: str = "figures_memoire", dry_run: bool = False) -> None:
    """
    Collecte toutes les figures générées dans un dossier unique.

    Parcourt le run courant et les sous-dossiers connus pour rassembler
    tous les .png et .pdf produits par le pipeline dans out/.

    Utile pour l'intégration LaTeX : \includegraphics{figures_memoire/...}
    """
    if dry_run:
        _print(f"🖼   [DRY-RUN] Figures ignorées → {out}/")
        return

    from config import get_current_run

    try:
        run_dir = get_current_run()
    except RuntimeError:
        safe_exit("Aucun run courant — lancez d'abord : make new-run && make analysis")

    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)

    collected = 0
    for ext in ("*.png", "*.pdf"):
        for src in run_dir.rglob(ext):
            # Préfixe avec le sous-dossier parent pour éviter les collisions
            parent = src.parent.name
            dst_name = f"{parent}__{src.name}" if parent != run_dir.name else src.name
            shutil.copy2(src, dest / dst_name)
            collected += 1

    _print(f"✔  {collected} figures collectées → {dest}/")


# =========================================================
# PIPELINE COMPLET DU MÉMOIRE
# =========================================================

def cmd_thesis(
    refresh:        bool = False,
    exclude_single: bool = True,
    dry_run:        bool = False,
) -> None:
    """
    Pipeline complet du mémoire : sync → new-run → analysis → regression → perception.

    Équivaut à :
        make sync
        make new-run
        make analysis
        make regression FEATURE_SET=all
        make perception
        make perc-space
        make figures
    """
    if dry_run:
        _print("📖  [DRY-RUN] Pipeline thèse — séquence complète")
        for cmd in ["sync", "new-run", "analysis (mode=groove)", "regression-all",
                    "perception", "perc-space", "figures"]:
            _info(f"  → {cmd}")
        return

    _print("📖  Pipeline du mémoire", style="bold blue")

    cmd_sync(dry_run=False)
    cmd_new_run()
    cmd_analysis(mode="groove", dry_run=False)
    cmd_regression_all(refresh=False, check_db=False,
                       exclude_single=exclude_single, dry_run=False)
    cmd_perception(refresh=False, dry_run=False)
    cmd_perception_space(refresh=False, dry_run=False)
    cmd_figures(out="figures_memoire", dry_run=False)

    _print("📖  Pipeline du mémoire terminé", style="bold green")


# =========================================================
# DOCTOR
# =========================================================

def cmd_doctor() -> None:
    from perception.check_supabase import check_supabase
    from utils.env_check import run_env_check

    _print("🩺  Diagnostics…", style="cyan")
    ok_db  = check_supabase(refresh=False, verbose=True)
    ok_env = run_env_check()

    if ok_db and ok_env:
        _print("✔  Tous les diagnostics passés")
    else:
        _error("Certains diagnostics ont échoué — voir ci-dessus")
        sys.exit(1)


# =========================================================
# ARGUMENT PARSER
# =========================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="groove",
        description="🎧  Groove Experiment — CLI du mémoire",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
══════════════════════════════════════════════════════════
 PIPELINE DU MÉMOIRE (ordre recommandé)
══════════════════════════════════════════════════════════
 §4.1  Génération des stimuli
   python cli.py --generate                    → MIDI + MP3 + metadata.csv
   python cli.py --validate                    → figures validation §4.1.4
   python cli.py --preview                     → 3 stimuli de référence

 §4.2  Collecte des données
   python cli.py --serve                       → serveur expérience web
   python cli.py --sync                        → Supabase → responses.csv

 §5    Espace latent
   python cli.py --new-run                     → dossier run horodaté
   python cli.py --analysis                    → embeddings + UMAP + clusters

 §6    Modélisation statistique
   python cli.py --regression-all --refresh    → Ridge · EN · SVR · RF · LMM
   python cli.py --perception --refresh        → alignement Ridge
   python cli.py --perception-space            → ICC · Mantel · géométrie

 Pipeline complet
   python cli.py --thesis                      → sync + analysis + modèles + figures
   python cli.py --figures                     → collecte figures → figures_memoire/

 Utilitaires
   python cli.py --status                      → état du système
   python cli.py --doctor                      → diagnostic Supabase + env
   python cli.py --dry-run --thesis            → simulation sans écriture

══════════════════════════════════════════════════════════
 ARCHITECTURE DES DONNÉES
══════════════════════════════════════════════════════════
 fetch_ratings()            FORMAT LONG   (1 ligne / réponse)
                            → --perception-space (ICC, variabilité)

 load_ratings_df()          FORMAT AGRÉGÉ (1 ligne / stimulus)
                            → --regression, --perception (Ridge)

 load_perceptual_dataset()  FORMAT JOINT  (stimuli × ratings)
                            → --perception (features acoustiques)

══════════════════════════════════════════════════════════
 STIMULI À RÉPONSE UNIQUE
══════════════════════════════════════════════════════════
 Par défaut, les stimuli avec n=1 réponse sont exclus de la
 régression (groove_std=0.0 artificiel, pas de consensus).
 Pour reproduire le run v1 :
   python cli.py --regression --include-single
══════════════════════════════════════════════════════════
""",
    )

    # ── §4.1 GÉNÉRATION ──────────────────────────────────────────────────────
    g = parser.add_argument_group("§4.1  Génération des stimuli")
    g.add_argument("--generate",   action="store_true",
                   help="Génère MIDI + MP3 + metadata.csv")
    g.add_argument("--validate",   action="store_true",
                   help="Figures de validation du générateur (§4.1.4)")
    g.add_argument("--preview",    action="store_true",
                   help="3 stimuli de référence pour écoute comparative")
    g.add_argument("--seed",       type=int, default=42,   metavar="N",
                   help="Graine aléatoire (défaut : 42)")
    g.add_argument("--repeats",    type=int, default=None, metavar="N",
                   help="Répétitions par condition (écrase config.py)")
    g.add_argument("--skip-audio", action="store_true",
                   help="Ignorer le rendu audio (MIDI seulement)")
    g.add_argument("--force", action="store_true",
                   help="Ignore le verrou .groove_locked (DANGER)")

    # ── §4.2 COLLECTE ────────────────────────────────────────────────────────
    g = parser.add_argument_group("§4.2  Collecte des données")
    g.add_argument("--serve",  action="store_true",
                   help="Démarre le serveur FastAPI de l'expérience")
    g.add_argument("--sync",   action="store_true",
                   help="Supabase → responses.csv (cache local)")

    # ── §5 ANALYSE ───────────────────────────────────────────────────────────
    g = parser.add_argument_group("§5  Espace latent")
    g.add_argument("--new-run",    action="store_true",
                   help="Crée un dossier de run horodaté")
    g.add_argument("--analysis",   action="store_true",
                   help="Pipeline embeddings + UMAP + clustering + export")
    g.add_argument("--analysis-only", action="store_true",
                   help="Alias de --analysis (rétro-compat)")
    g.add_argument(
        "--analysis-mode",
        default="groove",
        choices=["full", "groove", "full_with_clustering", "audio"],
        metavar="MODE",
        help="Mode du pipeline d'analyse (défaut : groove)",
    )
    g.add_argument("--steps", nargs="+", metavar="STEP",
                   help="Steps manuels (écrase --analysis-mode)")

    # ── §6 MODÉLISATION ──────────────────────────────────────────────────────
    g = parser.add_argument_group("§6  Modélisation statistique")
    g.add_argument("--regression",     action="store_true",
                   help="Régression groove (1 feature set)")
    g.add_argument("--regression-all", action="store_true",
                   help="Régression groove (design + acoustic + all)")
    g.add_argument(
        "--feature-set",
        default="all",
        choices=["design", "acoustic", "all", "interactions", "predictability"],
        metavar="FS",
        help="Feature set pour --regression (défaut : all)",
    )
    g.add_argument(
        "--include-single",
        action="store_true",
        default=False,
        help="Inclut les stimuli n=1 dans la régression (désactive exclude_single)",
    )
    g.add_argument("--perception",       action="store_true",
                   help="Alignement Ridge espace latent → groove_mean")
    g.add_argument("--perception-space", action="store_true",
                   help="ICC · Test de Mantel · géométrie locale k-NN")
    g.add_argument("--no-check-db",      action="store_true",
                   help="Ignore la vérification Supabase avant régression")

    # ── PIPELINE THÈSE ───────────────────────────────────────────────────────
    g = parser.add_argument_group("Pipeline complet")
    g.add_argument("--thesis",  action="store_true",
                   help="Pipeline thèse complet (sync+analysis+modèles+figures)")
    g.add_argument("--figures", action="store_true",
                   help="Collecte toutes les figures → figures_memoire/")
    g.add_argument("--figures-out", default="figures_memoire", metavar="DIR",
                   help="Dossier de destination pour --figures (défaut: figures_memoire)")

    # ── INFRA ────────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Infrastructure")
    g.add_argument("--refresh", action="store_true",
                   help="Force le re-fetch Supabase (ignore le cache local)")

    # ── MAINTENANCE ──────────────────────────────────────────────────────────
    g = parser.add_argument_group("Maintenance")
    g.add_argument(
        "--clean",
        nargs="*",
        choices=["all", "outputs", "metadata", "responses", "analysis", "cache"],
        metavar="TARGET",
        help="Nettoie les artefacts (all | outputs | metadata | responses | analysis | cache)",
    )
    g.add_argument("--status",  action="store_true",  help="État du système")
    g.add_argument("--doctor",  action="store_true",  help="Diagnostic Supabase + environnement")
    g.add_argument("--dry-run", action="store_true",  help="Simulation — rien n'est écrit")

    return parser


_ACTION_FLAGS = {
    "generate", "validate", "preview",
    "serve", "sync",
    "new_run", "analysis", "analysis_only",
    "regression", "regression_all",
    "perception", "perception_space",
    "thesis", "figures",
}


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    dry    = args.dry_run

    # exclude_single = True par défaut, False si --include-single
    exclude_single = not args.include_single

    if dry:
        _print("🔍  DRY-RUN — rien ne sera écrit ni exécuté", style="bold yellow")

    # ── Commandes sans pipeline ──────────────────────────────────────────────
    if args.status:
        cmd_status()
        return

    if args.doctor:
        cmd_doctor()
        return

    if args.clean is not None:
        cmd_clean(args.clean or ["all"], dry_run=dry)
        return

    # Aucune action demandée
    if not any(getattr(args, f, False) for f in _ACTION_FLAGS):
        parser.print_help()
        return

    # ── §4.1 Génération ──────────────────────────────────────────────────────
    if args.generate:
        ensure_data_dirs()
        cmd_generate(
            seed=args.seed,
            n_repeats=args.repeats,
            skip_audio=args.skip_audio,
            force=args.force,
            dry_run=dry,
        )

    if args.validate:
        cmd_validate(dry_run=dry)

    if args.preview:
        ensure_data_dirs()
        cmd_preview(seed=args.seed, dry_run=dry)

    # ── §4.2 Collecte ────────────────────────────────────────────────────────
    if args.serve:
        cmd_serve(dry_run=dry)
        return  # bloquant — on s'arrête ici

    if args.sync:
        cmd_sync(dry_run=dry)

    # ── §5 Analyse ───────────────────────────────────────────────────────────
    if args.new_run:
        cmd_new_run()

    if args.analysis or args.analysis_only:
        cmd_analysis(mode=args.analysis_mode, steps=args.steps, dry_run=dry)

    # ── §6 Modélisation ──────────────────────────────────────────────────────
    if args.regression:
        cmd_regression(
            feature_set=args.feature_set,
            refresh=args.refresh,
            check_db=not args.no_check_db,
            exclude_single=exclude_single,
            dry_run=dry,
        )

    if args.regression_all:
        cmd_regression_all(
            refresh=args.refresh,
            check_db=not args.no_check_db,
            exclude_single=exclude_single,
            dry_run=dry,
        )

    if args.perception:
        cmd_perception(refresh=args.refresh, dry_run=dry)

    if args.perception_space:
        cmd_perception_space(refresh=args.refresh, dry_run=dry)

    # ── Pipeline thèse ───────────────────────────────────────────────────────
    if args.thesis:
        cmd_thesis(
            refresh=args.refresh,
            exclude_single=exclude_single,
            dry_run=dry,
        )

    if args.figures:
        cmd_figures(out=args.figures_out, dry_run=dry)

    _done()


if __name__ == "__main__":
    main()