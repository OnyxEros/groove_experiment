import argparse
import sys
import shutil
from pathlib import Path

# =========================================================
# CONFIG (SAFE IMPORTS ONLY)
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
# UTILS
# =========================================================

def log(msg):
    print(f"\n{msg}\n")


def safe_exit(msg, code=1):
    print(f"\n❌ {msg}\n")
    sys.exit(code)


# =========================================================
# CLEAN (NO HEAVY IMPORTS)
# =========================================================

def clean():
    log("🧹 Cleaning outputs...")

    for d in [MIDI_DIR, WAV_DIR, MP3_DIR, PREVIEW_DIR]:
        if d.exists():
            shutil.rmtree(d)

    if METADATA_PATH.exists():
        METADATA_PATH.unlink()

    log("✔ Clean done")


# =========================================================
# GENERATION PIPELINE
# =========================================================

def run_generation(seed, n_repeats, skip_audio=False):

    from groove.generator import run_experiment
    from audio.midi_export import export_all
    from audio.mp3 import convert_all, build_audio_map

    ensure_data_dirs()

    log("🎛️ Generating stimuli...")
    df, stim_cache = run_experiment(seed=seed, n_repeats=n_repeats)

    log("🎼 Exporting MIDI...")
    export_all(df, stim_cache, out_dir=MIDI_DIR)

    if not skip_audio:
        log("🎧 Rendering audio...")

        if not Path(SOUNDFONT_PATH).exists():
            safe_exit(f"SoundFont not found: {SOUNDFONT_PATH}")

        convert_all(
            midi_root=MIDI_DIR,
            wav_root=WAV_DIR,
            mp3_root=MP3_DIR,
            soundfont=str(SOUNDFONT_PATH),
        )

    log("📊 Building dataset...")
    df = build_audio_map(df, mp3_root=MP3_DIR)

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(METADATA_PATH, index=False)

    log(f"✔ Generation complete → {len(df)} stimuli")

    return df


# =========================================================
# ANALYSIS (STRICT READ ONLY)
# =========================================================

def run_analysis(steps=None, mode="audio"):

    from analysis.core.run import run_analysis as engine_run

    print("🧠 Running analysis engine...")

    return engine_run(
        mode=mode,
        steps=steps,
        save=True,
        seed=42
    )

# =========================================================
# SUPABASE SYNC (LAZY IMPORT)
# =========================================================

def sync_supabase():

    from backend.dataset import load_dataset
    from infra.supabase_client import get_supabase

    log("☁️ Syncing to Supabase...")

    df = load_dataset()
    supabase = get_supabase()

    supabase.table("stimuli").upsert(
        df.to_dict(orient="records")
    ).execute()

    log(f"✔ Synced {len(df)} rows")


# =========================================================
# REGRESSION (LAZY)
# =========================================================

def run_regression_step():

    from regression.run import run_regression

    log("📈 Running regression...")
    return run_regression()


# =========================================================
# PERCEPTION (LAZY + ISOLATED)
# =========================================================

def run_perception():

    from backend.dataset import load_dataset
    import pandas as pd
    from perception.loader import load_perceptual_dataset
    from perception.alignment import fit_alignment
    from perception.metrics import cluster_perception_diff

    log("🧠 Running perception pipeline...")

    Z, stimulus_ids = load_dataset(return_embeddings=True)

    df = pd.DataFrame(Z, columns=["z1", "z2", "z3"])
    df["stimulus_id"] = stimulus_ids

    df = load_perceptual_dataset(df)

    Z = df[["z1", "z2", "z3"]].values
    ratings = df["rating"].values

    model, score = fit_alignment(Z, ratings)

    print("\n📊 Perceptual alignment R²:", score)

    if "cluster" in df.columns:
        scores = cluster_perception_diff(df["cluster"].values, ratings)
        print("\n📦 Cluster perception:", scores)

    return model, score


# =========================================================
# PREVIEW
# =========================================================

def preview(seed=42):

    from groove.generator import Grid, Voices, MicroTiming, Stimulus
    from audio.midi_export import export_all
    from audio.mp3 import convert_all

    import numpy as np
    import pandas as pd

    log("🎧 Preview mode...")

    grid = Grid()
    voices = Voices(grid, seed=seed)
    rng = np.random.default_rng(seed)
    micro = MicroTiming(rng, grid.step_duration)

    stim_builder = Stimulus(voices, micro)

    configs = [
        {"name": "baseline", "S_mv": 0, "D_mv": 1, "E": 0.0},
        {"name": "swing", "S_mv": 0, "D_mv": 1, "E": 0.5},
        {"name": "syncopated", "S_mv": 2, "D_mv": 1, "E": 0.0},
    ]

    rows = []
    cache = {}

    for i, cfg in enumerate(configs):
        stim = stim_builder.build(cfg, seed + i)
        cache[i] = stim

        rows.append({
            "id": i,
            "label": cfg["name"],
            **cfg
        })

    df = pd.DataFrame(rows)

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    export_all(df, cache, out_dir=PREVIEW_DIR)

    convert_all(
        midi_root=PREVIEW_DIR,
        wav_root=PREVIEW_DIR / "wav",
        mp3_root=PREVIEW_DIR / "mp3",
        soundfont=str(SOUNDFONT_PATH),
    )

    log("✔ Preview ready")


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser("🎧 Groove CLI (PRO refactored)")

    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--analysis", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--preview", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--skip-audio", action="store_true")

    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--regression", action="store_true")
    parser.add_argument("--perception", action="store_true")

    parser.add_argument("--analysis-mode", default="audio", choices=["full","audio", "groove"])
    parser.add_argument("--steps", nargs="+")

    parser.add_argument("--clean", action="store_true")

    args = parser.parse_args()

    # =====================================================
    # CLEAN (NO IMPORT SIDE EFFECTS)
    # =====================================================
    if args.clean:
        clean()
        return

    # =====================================================
    # PREVIEW
    # =====================================================
    if args.preview:
        ensure_data_dirs()
        preview(seed=args.seed)
        return

    # =====================================================
    # GENERATION ONLY
    # =====================================================
    if args.generate:
        run_generation(
            seed=args.seed,
            n_repeats=args.repeats,
            skip_audio=args.skip_audio
        )

    # =====================================================
    # ANALYSIS ONLY (READ-ONLY)
    # =====================================================
    if args.analysis or args.analysis_only:
        run_analysis(
            steps=args.steps,
            mode=args.analysis_mode
        )

    # =====================================================
    # OPTIONAL STEPS
    # =====================================================
    if args.sync:
        sync_supabase()

    if args.regression:
        run_regression_step()

    if args.perception:
        run_perception()

    print("\n🔥 DONE\n")


if __name__ == "__main__":
    main()