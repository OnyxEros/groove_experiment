"""
cli.py
======
Point d'entrée unique du système Groove Experiment.

Design :
    - Imports lourds uniquement à l'intérieur des fonctions (démarrage rapide)
    - Chaque commande est une fonction autonome avec dry_run natif
    - Le nothing-check est exhaustif (tous les flags listés)
    - Les erreurs sont catchées proprement avec safe_exit
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

# =========================================================
# RICH  (optionnel — pip install rich)
# =========================================================
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    _RICH = True
except ImportError:
    _RICH = False

# =========================================================
# CONFIG  (imports légers uniquement)
# =========================================================
from config import (
    ANALYSIS_DIR,
    METADATA_PATH,
    MIDI_DIR,
    MP3_DIR,
    PREVIEW_DIR,
    RESP_FILE,
    SOUNDFONT_PATH,
    WAV_DIR,
    ensure_data_dirs,
)

# =========================================================
# CONSOLE
# =========================================================

_console = Console() if _RICH else None


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


def _done() -> None:
    if _RICH:
        _console.print("\n[bold green]🔥  DONE[/bold green]\n")
    else:
        print("\n🔥  DONE\n")


def safe_exit(msg: str, code: int = 1) -> None:
    _error(msg)
    sys.exit(code)


# =========================================================
# STEP CONTEXT MANAGER  (spinner + timer)
# =========================================================

@contextmanager
def step(label: str, dry_run: bool = False):
    """Affiche un spinner pendant l'exécution d'une étape."""
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
                _console.print(
                    f"  ❌ [red]{label}[/red] failed after {elapsed:.1f}s"
                )
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
        _console.print(
            f"  ✔  [green]{label}[/green] [dim]({elapsed:.1f}s)[/dim]"
        )
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
    for pkg in ["numpy", "pandas", "scipy", "sklearn", "umap", "shap"]:
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
    """Affiche l'état du système (répertoires, dépendances, cache)."""
    if not _RICH:
        print("Status requires `rich`.  Install with: pip install rich")
        return

    table = Table(
        title="🎧  Groove System Status",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Component",    style="cyan",  min_width=18)
    table.add_column("Path / Info",  style="dim",   min_width=40)
    table.add_column("Status",       justify="center")

    # ── Répertoires ──────────────────────────────────────
    dirs = {
        "MIDI dir":    MIDI_DIR,
        "WAV dir":     WAV_DIR,
        "MP3 dir":     MP3_DIR,
        "Preview dir": PREVIEW_DIR,
        "Analysis dir":ANALYSIS_DIR,
        "Metadata":    METADATA_PATH,
        "SoundFont":   Path(SOUNDFONT_PATH),
    }
    for name, path in dirs.items():
        exists = path.exists()
        badge  = "[green]✔[/green]" if exists else "[red]✗[/red]"
        extra  = ""
        if exists and path.is_dir():
            n     = sum(1 for _ in path.rglob("*") if _.is_file())
            extra = f" ({n} files)"
        elif exists and path.is_file():
            size  = path.stat().st_size
            extra = f" ({size/1024:.0f} KB)"
        table.add_row(name, str(path) + extra, badge)

    # ── Réponses Supabase ────────────────────────────────
    resp = Path(RESP_FILE)
    if resp.exists():
        try:
            import pandas as pd
            n = len(pd.read_csv(resp))
            table.add_row(
                "responses.csv", str(resp),
                f"[green]✔  ({n} rows)[/green]",
            )
        except Exception:
            table.add_row("responses.csv", str(resp), "[yellow]⚠  unreadable[/yellow]")
    else:
        table.add_row("responses.csv", str(resp), "[dim]–  (not fetched)[/dim]")

    # ── Dernier run d'analyse ────────────────────────────
    if ANALYSIS_DIR.exists():
        runs = sorted(ANALYSIS_DIR.iterdir())
        if runs:
            latest = runs[-1]
            table.add_row("Latest run", str(latest.name), "[green]✔[/green]")

    # ── Dépendances Python ───────────────────────────────
    table.add_section()
    for pkg, ok in _check_deps().items():
        badge = "[green]✔[/green]" if ok else "[yellow]–  (optional)[/yellow]"
        table.add_row(pkg, f"import {pkg}", badge)

    _console.print(table)


# =========================================================
# CLEAN
# =========================================================

def cmd_clean(targets: list[str], dry_run: bool = False) -> None:
    """Nettoie les cibles spécifiées."""
    if not targets:
        targets = ["all"]

    _print(
        f"🧹  Clean targets: {targets}" + (" [DRY-RUN]" if dry_run else ""),
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
            _warn(f"Unknown clean target: {target!r} — skipped")
            continue
        if dry_run:
            _print(f"  [DRY-RUN] would clean: {target}", style="dim")
        else:
            dispatch[target]()

    _print("✔  Clean done")


def _clean_outputs() -> None:
    for d in [MIDI_DIR, WAV_DIR, MP3_DIR, PREVIEW_DIR]:
        if d.exists():
            shutil.rmtree(d)
            _print(f"  removed {d}", style="dim")


def _clean_metadata() -> None:
    if METADATA_PATH.exists():
        METADATA_PATH.unlink()
        _print(f"  removed {METADATA_PATH}", style="dim")


def _clean_responses() -> None:
    resp = Path(RESP_FILE)
    if resp.exists():
        resp.unlink()
        _print(f"  removed {resp}", style="dim")
    else:
        _warn("No local responses cache to remove.")


def _clean_analysis(subdirs: list[str] | None = None) -> None:
    if not ANALYSIS_DIR.exists():
        return
    if subdirs is None:
        shutil.rmtree(ANALYSIS_DIR)
        _print(f"  removed {ANALYSIS_DIR}", style="dim")
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
            _warn(f"Could not delete {d}: {exc}")
    for f in Path(".").rglob("*.pyc"):
        try:
            f.unlink()
            removed += 1
        except Exception:
            pass
    _print(f"  removed {removed} cache entries", style="dim")


# =========================================================
# GENERATION
# =========================================================

def cmd_generate(
    seed:       int,
    n_repeats:  int | None,
    skip_audio: bool = False,
    dry_run:    bool = False,
) -> None:
    """Génère les stimuli, les exports MIDI, rend l'audio et sauve metadata.csv."""
    if dry_run:
        _print("🎛️   [DRY-RUN] Generation pipeline — nothing will be written")
        for label in ["run_experiment", "export MIDI", "render audio (WAV→MP3)", "build dataset"]:
            _print(f"  → {label}", style="dim")
        return

    from groove.generator import run_experiment
    from audio.midi_export import export_all
    from audio.mp3 import convert_all, build_audio_map

    ensure_data_dirs()

    with step("Generate stimuli"):
        df, stim_cache = run_experiment(seed=seed, n_repeats=n_repeats)
        _print(f"  {len(df)} stimuli generated", style="dim")

    with step("Export MIDI"):
        export_all(df, stim_cache, out_dir=MIDI_DIR)

    if skip_audio:
        _warn("Audio rendering skipped (--skip-audio)")
    else:
        if not _check_soundfont():
            safe_exit(
                f"SoundFont not found: {SOUNDFONT_PATH}\n"
                "Use --skip-audio to bypass audio rendering."
            )
        with step("Render audio  (WAV → MP3)"):
            convert_all(
                midi_root=MIDI_DIR,
                wav_root=WAV_DIR,
                mp3_root=MP3_DIR,
                soundfont=str(SOUNDFONT_PATH),
            )

    with step("Build dataset"):
        df = build_audio_map(df, mp3_root=MP3_DIR)
        METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(METADATA_PATH, index=False)

    _print(f"✔  Generation complete — {len(df)} stimuli → {METADATA_PATH}")


# =========================================================
# PREVIEW
# =========================================================

def cmd_preview(seed: int = 42, dry_run: bool = False) -> None:
    """Génère 3 stimuli de preview (baseline, swing, syncopated)."""
    if dry_run:
        _print("🎧  [DRY-RUN] Preview skipped")
        return

    import numpy as np
    import pandas as pd
    from groove.generator import Grid, MicroTiming, Stimulus, Voices
    from audio.midi_export import export_all
    from audio.mp3 import convert_all

    configs = [
        {"name": "baseline",   "S_mv": 0, "D_mv": 1, "E": 0.0},
        {"name": "swing",      "S_mv": 0, "D_mv": 1, "E": 0.5},
        {"name": "syncopated", "S_mv": 2, "D_mv": 1, "E": 0.0},
    ]

    with step("Build preview stimuli"):
        grid    = Grid()
        voices  = Voices(grid, seed=seed)
        micro   = MicroTiming(np.random.default_rng(seed), grid.step_duration)
        builder = Stimulus(voices, micro)

        rows, cache = [], {}
        for i, cfg in enumerate(configs):
            stim = builder.build(cfg, seed + i)
            cache[i] = stim
            rows.append({"id": i, "label": cfg["name"], **cfg})

        df = pd.DataFrame(rows)
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    with step("Export preview MIDI"):
        export_all(df, cache, out_dir=PREVIEW_DIR)

    with step("Render preview audio"):
        convert_all(
            midi_root=PREVIEW_DIR,
            wav_root=PREVIEW_DIR / "wav",
            mp3_root=PREVIEW_DIR / "mp3",
            soundfont=str(SOUNDFONT_PATH),
        )

    _print(f"✔  Preview ready → {PREVIEW_DIR / 'mp3'}")


# =========================================================
# ANALYSIS
# =========================================================

def cmd_analysis(
    mode:    str = "audio",
    steps:   list[str] | None = None,
    dry_run: bool = False,
) -> None:
    """Lance le pipeline d'analyse (embeddings, clustering, viz…)."""
    if dry_run:
        _print(f"🧠  [DRY-RUN] Analysis — mode={mode}, steps={steps or 'default'}")
        return

    from analysis.core.run import run_analysis as _engine

    with step(f"Analysis engine  [mode={mode}]"):
        _engine(mode=mode, steps=steps, save=True, seed=42)


# =========================================================
# SYNC  (Supabase → cache local)
# =========================================================

def cmd_sync(dry_run: bool = False) -> None:
    """
    Fetch les réponses depuis Supabase et écrit data/responses.csv.
    N'écrit JAMAIS vers Supabase.
    """
    if dry_run:
        _print("☁️   [DRY-RUN] Fetch Supabase → local cache")
        return

    import pandas as pd
    from infra.supabase_client import fetch_responses

    with step("Fetch responses ← Supabase"):
        data = fetch_responses()
        if not data:
            safe_exit("Aucune réponse dans Supabase (table 'responses' vide).")
        df = pd.DataFrame(data)

    with step("Write local cache"):
        cache_path = Path(RESP_FILE)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)

    _print(f"✔  {len(df)} réponses → {RESP_FILE}")


# =========================================================
# REGRESSION
# =========================================================

def cmd_regression(
    feature_set: str  = "all",
    refresh:     bool = False,
    check_db:    bool = True,
    dry_run:     bool = False,
) -> None:
    """Lance la régression groove (Ridge + RandomForest) sur un feature set."""
    if dry_run:
        _print(f"📈  [DRY-RUN] Regression — features={feature_set}")
        return

    from regression.run import run_regression

    with step(f"Regression  [features={feature_set}]"):
        result = run_regression(
            feature_set=feature_set,
            refresh=refresh,
            check_db=check_db,
        )

    r2 = result.get("best_r2")
    best = result.get("best_model", "?")
    _print(
        f"✔  Regression done — best={best}"
        + (f"  R²={r2:.3f}" if r2 is not None else ""),
    )


def cmd_regression_all(
    refresh:  bool = False,
    check_db: bool = True,
    dry_run:  bool = False,
) -> None:
    """Lance les 3 feature sets en séquence (pour le mémoire)."""
    for fs in ("design", "acoustic", "all"):
        cmd_regression(
            feature_set=fs,
            refresh=(refresh and fs == "design"),  # re-fetch une seule fois
            check_db=(check_db and fs == "design"),
            dry_run=dry_run,
        )


# =========================================================
# PERCEPTION
# =========================================================

def cmd_perception(refresh: bool = False, dry_run: bool = False) -> None:
    """Aligne l'espace latent avec les ratings perceptifs."""
    if dry_run:
        _print("🧠  [DRY-RUN] Perception alignment skipped")
        return

    import pandas as pd
    from perception.alignment import fit_alignment
    from perception.loader import load_perceptual_dataset
    from perception.metrics import cluster_perception_diff

    with step("Load perceptual dataset"):
        meta = pd.read_csv(METADATA_PATH)
        if "stim_id" not in meta.columns and "id" in meta.columns:
            meta = meta.rename(columns={"id": "stim_id"})
        df = load_perceptual_dataset(embedding_df=meta, refresh=refresh)

    with step("Fit alignment  (Ridge latent → groove)"):
        feature_cols = [c for c in ["D", "I", "V", "S_real", "E_real"] if c in df.columns]
        model, score = fit_alignment(df[feature_cols].values, df["groove_mean"].values)

    _print(f"📊  Perceptual alignment R² = {score:.4f}")

    if "cluster" in df.columns:
        scores = cluster_perception_diff(df["cluster"].values, df["groove_mean"].values)
        _print(f"📦  Cluster groove means : {scores}")


def cmd_perception_space(refresh: bool = False, dry_run: bool = False) -> None:
    """Analyse géométrique du groove dans l'espace latent UMAP."""
    if dry_run:
        _print("🧠  [DRY-RUN] Perception space skipped")
        return

    import pandas as pd
    from perception_space.run import run_perception_space

    with step("Load metadata"):
        df = pd.read_csv(METADATA_PATH)
        if "stim_id" not in df.columns and "id" in df.columns:
            df = df.rename(columns={"id": "stim_id"})

    with step("Resolve latest analysis run"):
        from analysis.io.run_resolver import get_latest_run
        try:
            run_dir = get_latest_run()
        except (ValueError, FileNotFoundError) as exc:
            safe_exit(
                f"No analysis run found — launch `make analysis` first.\n({exc})"
            )

    with step("Perception space geometry"):
        run_perception_space(run_dir=run_dir, perception_data=df)

    _print("✔  Perception space computed")


# =========================================================
# DOCTOR  (diagnostic complet)
# =========================================================

def cmd_doctor() -> None:
    """Diagnostic Supabase + variables d'environnement."""
    from perception.check_supabase import check_supabase
    from utils.env_check import run_env_check  # type: ignore[import]

    _print("🩺  Running diagnostics…", style="cyan")
    ok_db  = check_supabase(refresh=False, verbose=True)
    ok_env = run_env_check()  # type: ignore[call-arg]

    if ok_db and ok_env:
        _print("✔  All checks passed")
    else:
        _error("Some checks failed — see above")
        sys.exit(1)


# =========================================================
# ARGUMENT PARSER
# =========================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="groove",
        description="🎧  Groove Experiment CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
──────────────────────────────────────────────────────────────
QUICK REFERENCE
──────────────────────────────────────────────────────────────
 Generate stimuli         python cli.py --generate
 Run full analysis        python cli.py --analysis --analysis-mode full
 Fetch Supabase data      python cli.py --sync
 Run all regressions      python cli.py --regression-all --refresh
 Run single regression    python cli.py --regression --feature-set design
 Perceptual alignment     python cli.py --perception --refresh
 Perception geometry      python cli.py --perception-space
 Full thesis pipeline     python cli.py --regression-all --perception --perception-space --refresh
 System status            python cli.py --status
 Supabase diagnostic      python cli.py --doctor
 Dry-run any command      python cli.py --generate --dry-run
──────────────────────────────────────────────────────────────
""",
    )

    # ── Pipeline ─────────────────────────────────────────
    g = parser.add_argument_group("Pipeline")
    g.add_argument("--generate",      action="store_true", help="Generate stimuli (MIDI + audio + metadata.csv)")
    g.add_argument("--analysis",      action="store_true", help="Run analysis pipeline (embeddings, clustering, viz)")
    g.add_argument("--analysis-only", action="store_true", help="Alias for --analysis")
    g.add_argument("--preview",       action="store_true", help="Generate 3 preview stimuli (baseline / swing / syncopated)")

    # ── Generation ───────────────────────────────────────
    g = parser.add_argument_group("Generation options")
    g.add_argument("--seed",       type=int, default=42,   metavar="N",  help="Master random seed (default: 42)")
    g.add_argument("--repeats",    type=int, default=None, metavar="N",  help="Repeats per condition (default: config.REPEATS)")
    g.add_argument("--skip-audio", action="store_true",                  help="Skip WAV/MP3 rendering (faster, no SoundFont needed)")

    # ── Analysis ─────────────────────────────────────────
    g = parser.add_argument_group("Analysis options")
    g.add_argument(
        "--analysis-mode",
        default="audio",
        choices=["full", "audio", "groove"],
        metavar="MODE",
        help="Pipeline mode: full | audio | groove  (default: audio)",
    )
    g.add_argument(
        "--steps",
        nargs="+",
        metavar="STEP",
        help="Override pipeline steps (e.g. embeddings projection clustering viz export)",
    )

    # ── Modelling ────────────────────────────────────────
    g = parser.add_argument_group("Modelling")
    g.add_argument("--regression",     action="store_true", help="Run regression for one feature set (see --feature-set)")
    g.add_argument("--regression-all", action="store_true", help="Run regression for ALL feature sets (design / acoustic / all) — thesis mode")
    g.add_argument(
        "--feature-set",
        default="all",
        choices=["design", "acoustic", "all"],
        metavar="FS",
        help="Feature set for --regression: design | acoustic | all  (default: all)",
    )
    g.add_argument("--perception",       action="store_true", help="Run perceptual alignment (latent space → groove ratings)")
    g.add_argument("--perception-space", action="store_true", help="Run geometric analysis of groove in UMAP latent space")
    g.add_argument("--no-check-db",      action="store_true", help="Skip Supabase connectivity check before regression (faster offline)")

    # ── Infra ────────────────────────────────────────────
    g = parser.add_argument_group("Infra")
    g.add_argument("--sync",    action="store_true", help="Fetch Supabase responses → data/responses.csv (read-only)")
    g.add_argument("--refresh", action="store_true", help="Force re-fetch from Supabase even if local cache exists")

    # ── Maintenance ──────────────────────────────────────
    g = parser.add_argument_group("Maintenance")
    g.add_argument(
        "--clean",
        nargs="*",
        choices=["all", "outputs", "metadata", "responses", "analysis", "cache"],
        metavar="TARGET",
        help="Clean targets: all | outputs | metadata | responses | analysis | cache",
    )
    g.add_argument("--status",  action="store_true", help="Print system status (dirs, deps, cache)")
    g.add_argument("--doctor",  action="store_true", help="Run Supabase + env diagnostic")
    g.add_argument("--dry-run", action="store_true", help="Show what would run without executing anything")

    return parser


# =========================================================
# MAIN
# =========================================================

# Flags qui comptent comme "quelque chose est spécifié"
_ACTION_FLAGS = {
    "generate", "analysis", "analysis_only",
    "preview",
    "regression", "regression_all",
    "perception", "perception_space",
    "sync",
}


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    dry    = args.dry_run

    if dry:
        _print("🔍  DRY-RUN — nothing will be written or executed", style="bold yellow")

    # ── Commandes standalone (retour immédiat) ────────────
    if args.status:
        cmd_status()
        return

    if args.doctor:
        cmd_doctor()
        return

    if args.clean is not None:
        cmd_clean(args.clean or ["all"], dry_run=dry)
        return

    if args.preview:
        ensure_data_dirs()
        cmd_preview(seed=args.seed, dry_run=dry)
        return

    # ── Rien de spécifié → aide ───────────────────────────
    if not any(getattr(args, f, False) for f in _ACTION_FLAGS):
        parser.print_help()
        return

    # ── Séquence pipeline ─────────────────────────────────

    # 1. Génération
    if args.generate:
        cmd_generate(
            seed=args.seed,
            n_repeats=args.repeats,
            skip_audio=args.skip_audio,
            dry_run=dry,
        )

    # 2. Analyse
    if args.analysis or args.analysis_only:
        cmd_analysis(
            mode=args.analysis_mode,
            steps=args.steps,
            dry_run=dry,
        )

    # 3. Sync Supabase
    if args.sync:
        cmd_sync(dry_run=dry)

    # 4. Régression (un feature set)
    if args.regression:
        cmd_regression(
            feature_set=args.feature_set,
            refresh=args.refresh,
            check_db=not args.no_check_db,
            dry_run=dry,
        )

    # 5. Régression (tous les feature sets — mode mémoire)
    if args.regression_all:
        cmd_regression_all(
            refresh=args.refresh,
            check_db=not args.no_check_db,
            dry_run=dry,
        )

    # 6. Alignement perceptif
    if args.perception:
        cmd_perception(refresh=args.refresh, dry_run=dry)

    # 7. Géométrie espace perceptif
    if args.perception_space:
        cmd_perception_space(refresh=args.refresh, dry_run=dry)

    _done()


if __name__ == "__main__":
    main()