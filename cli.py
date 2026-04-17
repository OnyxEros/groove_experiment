import argparse
import sys
from pathlib import Path
import shutil

from config import (
    ensure_data_dirs,
    MIDI_DIR,
    WAV_DIR,
    MP3_DIR,
    METADATA_PATH,
    SOUNDFONT_PATH,
)

from groove.generator import run_experiment
from audio.midi_export import export_all
from audio.mp3 import convert_all, build_audio_map

from analysis.pipelines.full_analysis import run_full_analysis

from backend.dataset import load_dataset
from backend.supabase_client import upsert_rows
from backend.dataset_store import sync_responses

from regression import load_dataset as load_reg_dataset
from regression import train_model, evaluate_model


# =========================================================
# UTILS
# =========================================================

def log(msg):
    print(f"\n{msg}\n")


def safe_exit(msg, code=1):
    print(f"\n❌ {msg}\n")
    sys.exit(code)


# =========================================================
# CLEAN
# =========================================================

def clean():
    log("🧹 Cleaning outputs...")

    for d in [MIDI_DIR, WAV_DIR, MP3_DIR]:
        if d.exists():
            shutil.rmtree(d)

    if METADATA_PATH.exists():
        METADATA_PATH.unlink()

    log("✔ Clean done")


# =========================================================
# PIPELINE STEPS
# =========================================================

def step_generate(seed, n_repeats):
    log("🎛️ Generating stimuli...")
    return run_experiment(seed=seed, n_repeats=n_repeats)


def step_midi(df, stim_cache):
    log("🎼 Exporting MIDI...")
    export_all(df, stim_cache, out_dir=MIDI_DIR)


def step_audio():
    log("🎧 Rendering audio...")

    if not Path(SOUNDFONT_PATH).exists():
        safe_exit(f"SoundFont not found: {SOUNDFONT_PATH}")

    convert_all(
        midi_root=MIDI_DIR,
        wav_root=WAV_DIR,
        mp3_root=MP3_DIR,
        soundfont=str(SOUNDFONT_PATH),
    )


def step_dataset(df):
    log("📊 Building dataset...")

    df = build_audio_map(df, mp3_root=MP3_DIR)

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(METADATA_PATH, index=False)

    log(f"✔ Dataset saved → {METADATA_PATH}")

    return df


def step_analysis():
    log("🧠 Running full analysis pipeline...")

    if not Path(MP3_DIR).exists():
        safe_exit("MP3 directory not found. Run full pipeline first.")

    return run_full_analysis(mp3_dir=MP3_DIR, save=True)


# =========================================================
# SUPABASE SYNC
# =========================================================

def step_sync_supabase():
    log("☁️ Syncing to Supabase...")

    df = load_dataset()

    rows = df.to_dict(orient="records")
    upsert_rows("responses", rows)

    log(f"✔ Synced {len(rows)} rows")


# =========================================================
# REGRESSION
# =========================================================

def step_regression():
    log("📈 Running regression...")

    df = load_reg_dataset()

    model = train_model(df)
    metrics = evaluate_model(model, df)

    print("\n📊 Regression metrics:")
    print(metrics)

    return model, metrics


# =========================================================
# FULL PIPELINE
# =========================================================

def run_full(seed=42, n_repeats=8, skip_audio=False):
    ensure_data_dirs()

    df, stim_cache = step_generate(seed, n_repeats)

    step_midi(df, stim_cache)

    if not skip_audio:
        step_audio()
    else:
        log("⚠️ Skipping audio rendering")

    df = step_dataset(df)

    log(f"✔ Pipeline complete → {len(df)} stimuli")

    return df


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="🎧 Groove Experiment CLI")

    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-audio", action="store_true")

    parser.add_argument("--analysis", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")

    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--regression", action="store_true")

    args = parser.parse_args()

    # =====================================================
    # CLEAN
    # =====================================================
    if args.clean:
        clean()
        return

    # =====================================================
    # ANALYSIS ONLY
    # =====================================================
    if args.analysis_only:
        log("📂 Loading existing dataset...")

        try:
            load_dataset()
        except Exception:
            safe_exit("Dataset not found. Run full pipeline first.")

        step_analysis()
        return

    # =====================================================
    # FULL PIPELINE
    # =====================================================
    df = run_full(
        seed=args.seed,
        n_repeats=args.repeats,
        skip_audio=args.skip_audio,
    )

    # =====================================================
    # OPTIONAL STEPS
    # =====================================================

    if args.analysis:
        step_analysis()

    if args.sync:
        step_sync_supabase()

    if args.regression:
        step_regression()

    print("\n🔥 READY\n")


# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":
    main()