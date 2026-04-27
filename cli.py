import argparse
import sys
import shutil
import time
import traceback
from pathlib import Path
from contextlib import contextmanager

# =========================================================
# RICH (optional — pip install rich)
# =========================================================
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    RICH = True
except ImportError:
    RICH = False

# =========================================================
# CONFIG (safe imports only — no heavy deps)
# =========================================================
from config import (
    ensure_data_dirs,
    MIDI_DIR,
    WAV_DIR,
    MP3_DIR,
    PREVIEW_DIR,
    METADATA_PATH,
    SOUNDFONT_PATH,
)

# =========================================================
# CONSOLE / LOGGING
# =========================================================

console = Console() if RICH else None


def log(msg: str, style: str = "bold green"):
    if RICH:
        console.print(f"\n{msg}\n", style=style)
    else:
        print(f"\n{msg}\n")


def log_error(msg: str):
    if RICH:
        console.print(f"\n❌ {msg}\n", style="bold red")
    else:
        print(f"\n❌ {msg}\n", file=sys.stderr)


def log_warn(msg: str):
    if RICH:
        console.print(f"⚠️  {msg}", style="yellow")
    else:
        print(f"⚠️  {msg}")


def safe_exit(msg: str, code: int = 1):
    log_error(msg)
    sys.exit(code)


@contextmanager
def step(label: str, dry_run: bool = False):
    """Context manager for timed pipeline steps with rich spinner."""
    if dry_run:
        if RICH:
            console.print(f"  [dim]DRY-RUN[/dim] [cyan]{label}[/cyan]")
        else:
            print(f"  [DRY-RUN] {label}")
        yield
        return

    start = time.perf_counter()
    if RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold cyan]{label}[/bold cyan]"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("", total=None)
            try:
                yield
            except Exception:
                elapsed = time.perf_counter() - start
                console.print(f"  ❌ [red]{label}[/red] failed after {elapsed:.1f}s")
                console.print_exception(show_locals=False)
                raise
    else:
        print(f"\n▶ {label}...")
        try:
            yield
        except Exception as e:
            elapsed = time.perf_counter() - start
            print(f"  FAILED after {elapsed:.1f}s: {e}")
            traceback.print_exc()
            raise

    elapsed = time.perf_counter() - start
    if RICH:
        console.print(f"  ✔ [green]{label}[/green] [dim]({elapsed:.1f}s)[/dim]")
    else:
        print(f"  ✔ {label} ({elapsed:.1f}s)")


# =========================================================
# PREFLIGHT CHECKS
# =========================================================

def check_soundfont() -> bool:
    ok = Path(SOUNDFONT_PATH).exists()
    if not ok:
        log_warn(f"SoundFont not found: {SOUNDFONT_PATH}")
    return ok


def check_dependencies() -> dict[str, bool]:
    checks = {}
    for pkg in ["numpy", "pandas", "scipy", "sklearn", "umap"]:
        try:
            __import__(pkg.replace("-", "_"))
            checks[pkg] = True
        except ImportError:
            checks[pkg] = False
    return checks


def print_status():
    if not RICH:
        print("Status requires `rich`. Install with: pip install rich")
        return

    table = Table(
        title="🎧 Groove System Status",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Component", style="cyan")
    table.add_column("Path / Package", style="dim")
    table.add_column("Status", justify="center")

    dirs = {
        "MIDI dir":   MIDI_DIR,
        "WAV dir":    WAV_DIR,
        "MP3 dir":    MP3_DIR,
        "Preview dir": PREVIEW_DIR,
        "Metadata":   METADATA_PATH,
        "SoundFont":  Path(SOUNDFONT_PATH),
    }

    for name, path in dirs.items():
        exists = path.exists()
        status = "[green]✔[/green]" if exists else "[red]✗[/red]"
        extra = ""
        if exists and path.is_dir():
            n = len(list(path.rglob("*")))
            extra = f" ({n} files)"
        table.add_row(name, str(path) + extra, status)

    deps = check_dependencies()
    for pkg, ok in deps.items():
        status = "[green]✔[/green]" if ok else "[yellow]–[/yellow]"
        table.add_row(pkg, f"import {pkg}", status)

    # cache local des réponses
    from config import RESP_FILE
    resp = Path(RESP_FILE)
    if resp.exists():
        import pandas as pd
        try:
            n = len(pd.read_csv(resp))
            table.add_row("responses.csv", str(resp), f"[green]✔ ({n} rows)[/green]")
        except Exception:
            table.add_row("responses.csv", str(resp), "[yellow]⚠ unreadable[/yellow]")
    else:
        table.add_row("responses.csv", str(resp), "[dim]–[/dim]")

    console.print(table)


# =========================================================
# CLEAN
# =========================================================

def clean_outputs():
    log("🧹 Cleaning outputs (MIDI / WAV / MP3 / PREVIEW)...")
    for d in [MIDI_DIR, WAV_DIR, MP3_DIR, PREVIEW_DIR]:
        if d.exists():
            shutil.rmtree(d)
            log(f"  removed {d}", style="dim")


def clean_metadata():
    log("🧹 Cleaning metadata...")
    if METADATA_PATH.exists():
        METADATA_PATH.unlink()


def clean_responses():
    """Supprime le cache local des réponses Supabase (pas la table cloud)."""
    from config import RESP_FILE
    path = Path(RESP_FILE)
    if path.exists():
        path.unlink()
        log(f"🧹 Removed local responses cache: {path}")
    else:
        log_warn("No local responses cache to remove.")


def clean_analysis(subdirs=None):
    log("🧹 Cleaning analysis...")
    analysis_dir = Path("data/analysis")
    if not analysis_dir.exists():
        return
    if subdirs is None:
        shutil.rmtree(analysis_dir)
    else:
        for sub in subdirs:
            p = analysis_dir / sub
            if p.exists():
                shutil.rmtree(p)


def clean_pycache():
    log("🧹 Cleaning __pycache__...")
    for pycache in Path(".").rglob("__pycache__"):
        try:
            shutil.rmtree(pycache)
        except Exception as e:
            log_warn(f"Could not delete {pycache}: {e}")


def clean(levels: list[str], dry_run: bool = False):
    if not levels:
        levels = ["all"]

    log(f"🧹 Clean targets: {levels}" + (" [DRY-RUN]" if dry_run else ""))

    dispatch = {
        "outputs":   clean_outputs,
        "metadata":  clean_metadata,
        "responses": clean_responses,
        "analysis":  clean_analysis,
        "cache":     clean_pycache,
    }

    targets = list(dispatch.keys()) if "all" in levels else [
        lv for lv in levels if lv in dispatch
    ]

    for target in targets:
        if dry_run:
            log(f"  [DRY-RUN] would clean: {target}", style="dim")
        else:
            dispatch[target]()

    log("✔ Clean done")


# =========================================================
# GENERATION
# =========================================================

def run_generation(
    seed: int,
    n_repeats,
    skip_audio: bool = False,
    dry_run: bool = False,
):
    if dry_run:
        log("🎛️  [DRY-RUN] Generation pipeline — no files will be written")
        for label in ["run_experiment", "export MIDI", "render audio", "build dataset"]:
            log(f"  → {label}", style="dim")
        return None

    from groove.generator import run_experiment
    from audio.midi_export import export_all
    from audio.mp3 import convert_all, build_audio_map

    ensure_data_dirs()

    with step("Generate stimuli"):
        df, stim_cache = run_experiment(seed=seed, n_repeats=n_repeats)
        log(f"  {len(df)} stimuli generated", style="dim")

    with step("Export MIDI"):
        export_all(df, stim_cache, out_dir=MIDI_DIR)

    if not skip_audio:
        if not check_soundfont():
            safe_exit(f"SoundFont not found: {SOUNDFONT_PATH}. Use --skip-audio to bypass.")
        with step("Render audio (WAV → MP3)"):
            convert_all(
                midi_root=MIDI_DIR,
                wav_root=WAV_DIR,
                mp3_root=MP3_DIR,
                soundfont=str(SOUNDFONT_PATH),
            )
    else:
        log_warn("Audio rendering skipped (--skip-audio)")

    with step("Build dataset"):
        df = build_audio_map(df, mp3_root=MP3_DIR)
        METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(METADATA_PATH, index=False)

    log(f"✔ Generation complete → {len(df)} stimuli saved to {METADATA_PATH}")
    return df


# =========================================================
# ANALYSIS
# =========================================================

def run_analysis(steps=None, mode: str = "audio", dry_run: bool = False):
    if dry_run:
        log(f"🧠 [DRY-RUN] Analysis — mode={mode}, steps={steps or 'default'}")
        return None

    from analysis.core.run import run_analysis as engine_run

    with step(f"Analysis engine [mode={mode}]"):
        result = engine_run(mode=mode, steps=steps, save=True, seed=42)

    return result


# =========================================================
# SUPABASE SYNC
# =========================================================

def sync_supabase(dry_run: bool = False):
    """
    Récupère les réponses depuis Supabase et les stocke en cache local.
    Aucun write vers Supabase (mode local-first).
    """

    import pandas as pd
    from pathlib import Path
    from infra.supabase_client import fetch_responses
    from config import RESP_FILE

    cache_path = Path(RESP_FILE)

    if dry_run:
        log("☁️  [DRY-RUN] Fetch Supabase → local cache (responses)")
        return

    with step("Fetch responses → Supabase"):
        data = fetch_responses()

        if not data:
            safe_exit("Aucune réponse trouvée dans Supabase (table 'responses' vide).")

        df = pd.DataFrame(data)

    with step("Write local cache"):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)

    log(f"✔ Fetched {len(df)} responses → {cache_path}")


# =========================================================
# REGRESSION
# =========================================================

def run_regression_step(
    feature_set: str = "all",
    refresh: bool = False,
    dry_run: bool = False,
):
    if dry_run:
        log("📈 [DRY-RUN] Regression skipped")
        return

    from regression.run import run_regression

    with step(f"Regression [features={feature_set}]"):
        result = run_regression(feature_set=feature_set, refresh=refresh)

    return result


# =========================================================
# PERCEPTION
# =========================================================

def run_perception(refresh: bool = False, dry_run: bool = False):
    if dry_run:
        log("🧠 [DRY-RUN] Perception pipeline skipped")
        return None, None

    import pandas as pd
    from perception.loader import load_perceptual_dataset
    from perception.alignment import fit_alignment
    from perception.metrics import cluster_perception_diff
    from config import METADATA_PATH

    with step("Load perceptual dataset"):
        meta = pd.read_csv(METADATA_PATH)
        if "stim_id" not in meta.columns and "id" in meta.columns:
            meta = meta.rename(columns={"id": "stim_id"})
        df = load_perceptual_dataset(embedding_df=meta, refresh=refresh)

    with step("Fit alignment"):
        feature_cols = [c for c in ["D", "I", "V", "S_real", "E_real"] if c in df.columns]
        Z = df[feature_cols].values
        ratings = df["groove_mean"].values
        model, score = fit_alignment(Z, ratings)

    log(f"📊 Perceptual alignment R²: {score:.4f}")

    if "cluster" in df.columns:
        scores = cluster_perception_diff(df["cluster"].values, ratings)
        log(f"📦 Cluster perception: {scores}")

    return model, score


# =========================================================
# PREVIEW
# =========================================================

def preview(seed: int = 42, dry_run: bool = False):
    if dry_run:
        log("🎧 [DRY-RUN] Preview skipped")
        return

    from groove.generator import Grid, Voices, MicroTiming, Stimulus
    from audio.midi_export import export_all
    from audio.mp3 import convert_all
    import numpy as np
    import pandas as pd

    configs = [
        {"name": "baseline",   "S_mv": 0, "D_mv": 1, "E": 0.0},
        {"name": "swing",      "S_mv": 0, "D_mv": 1, "E": 0.5},
        {"name": "syncopated", "S_mv": 2, "D_mv": 1, "E": 0.0},
    ]

    with step("Build preview stimuli"):
        grid = Grid()
        voices = Voices(grid, seed=seed)
        rng = np.random.default_rng(seed)
        micro = MicroTiming(rng, grid.step_duration)
        stim_builder = Stimulus(voices, micro)

        rows, cache = [], {}
        for i, cfg in enumerate(configs):
            stim = stim_builder.build(cfg, seed + i)
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

    log(f"✔ Preview ready → {PREVIEW_DIR / 'mp3'}")


# =========================================================
# ARGUMENT PARSER
# =========================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="groove",
        description="🎧 Groove Experiment CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py --generate --seed 42 --repeats 8
  python cli.py --analysis --analysis-mode full
  python cli.py --generate --analysis --sync
  python cli.py --regression --feature-set design
  python cli.py --perception --refresh
  python cli.py --clean all
  python cli.py --clean responses          # vide le cache local Supabase
  python cli.py --generate --dry-run
  python cli.py --status
        """,
    )

    # ── Pipeline ─────────────────────────────────────────
    pipeline = parser.add_argument_group("Pipeline")
    pipeline.add_argument("--generate",      action="store_true", help="Run generation pipeline")
    pipeline.add_argument("--analysis",      action="store_true", help="Run analysis pipeline")
    pipeline.add_argument("--analysis-only", action="store_true", help="Alias for --analysis")
    pipeline.add_argument("--preview",       action="store_true", help="Generate preview stimuli")

    # ── Generation ───────────────────────────────────────
    gen = parser.add_argument_group("Generation options")
    gen.add_argument("--seed",       type=int, default=42,   help="Random seed (default: 42)")
    gen.add_argument("--repeats",    type=int, default=None, help="Number of repeats per condition")
    gen.add_argument("--skip-audio", action="store_true",    help="Skip WAV/MP3 rendering")

    # ── Analysis ─────────────────────────────────────────
    ana = parser.add_argument_group("Analysis options")
    ana.add_argument(
        "--analysis-mode",
        default="audio",
        choices=["full", "audio", "groove"],
        help="Analysis mode (default: audio)",
    )
    ana.add_argument(
        "--steps",
        nargs="+",
        metavar="STEP",
        help="Custom steps: embeddings projection clustering metrics_view viz export",
    )

    # ── Modelling ────────────────────────────────────────
    mod = parser.add_argument_group("Modelling")
    mod.add_argument("--regression",  action="store_true", help="Run regression model")
    mod.add_argument(
        "--feature-set",
        default="all",
        choices=["design", "acoustic", "all"],
        help="Feature set for regression (default: all)",
    )
    mod.add_argument("--perception",  action="store_true", help="Run perceptual alignment")

    # ── Infra ────────────────────────────────────────────
    infra = parser.add_argument_group("Infra")
    infra.add_argument("--sync",    action="store_true", help="Sync stimuli metadata → Supabase")
    infra.add_argument("--refresh", action="store_true", help="Re-fetch réponses depuis Supabase (ignore le cache local)")

    # ── Maintenance ──────────────────────────────────────
    maint = parser.add_argument_group("Maintenance")
    maint.add_argument(
        "--clean",
        nargs="*",
        choices=["all", "outputs", "metadata", "responses", "analysis", "cache"],
        metavar="TARGET",
        help="Clean: all outputs metadata responses analysis cache",
    )
    maint.add_argument("--status",  action="store_true", help="Print system status")
    maint.add_argument("--dry-run", action="store_true", help="Show what would run, without executing")

    return parser


# =========================================================
# MAIN
# =========================================================

def main():
    parser = build_parser()
    args = parser.parse_args()
    dry = args.dry_run

    if dry:
        log("🔍 DRY-RUN mode — no files will be written", style="bold yellow")

    # ── Status ───────────────────────────────────────────
    if args.status:
        print_status()
        return

    # ── Clean ────────────────────────────────────────────
    if args.clean is not None:
        clean(args.clean or ["all"], dry_run=dry)
        return

    # ── Preview ──────────────────────────────────────────
    if args.preview:
        if not dry:
            ensure_data_dirs()
        preview(seed=args.seed, dry_run=dry)
        return

    # ── Nothing specified ────────────────────────────────
    nothing = not any([
        args.generate,
        args.analysis,
        args.analysis_only,
        args.sync,
        args.regression,
        args.perception,
    ])
    if nothing:
        parser.print_help()
        return

    # ── Generation ───────────────────────────────────────
    if args.generate:
        run_generation(
            seed=args.seed,
            n_repeats=args.repeats,
            skip_audio=args.skip_audio,
            dry_run=dry,
        )

    # ── Analysis ─────────────────────────────────────────
    if args.analysis or args.analysis_only:
        run_analysis(
            steps=args.steps,
            mode=args.analysis_mode,
            dry_run=dry,
        )

    # ── Sync ─────────────────────────────────────────────
    if args.sync:
        sync_supabase(dry_run=dry)

    # ── Regression ───────────────────────────────────────
    if args.regression:
        run_regression_step(
            feature_set=args.feature_set,
            refresh=args.refresh,
            dry_run=dry,
        )

    # ── Perception ───────────────────────────────────────
    if args.perception:
        run_perception(refresh=args.refresh, dry_run=dry)

    # ── Done ─────────────────────────────────────────────
    if RICH:
        console.print("\n[bold green]🔥 DONE[/bold green]\n")
    else:
        print("\n🔥 DONE\n")


if __name__ == "__main__":
    main()