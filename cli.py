import argparse
import shutil
from tqdm import tqdm

from groove.generator import run_experiment
from audio.midi_export import export_all
from audio.mp3 import convert_all, build_audio_map

from config import (
    ensure_data_dirs,
    MIDI_DIR,
    WAV_DIR,
    MP3_DIR,
    METADATA_PATH,
    SOUNDFONT_PATH
)

# =========================================================
# UTILS
# =========================================================

def log(msg):
    print(f"\n{msg}\n")


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

    log("✔ clean done")


# =========================================================
# STEPS
# =========================================================

def step_generate(seed, n_repeats):
    return run_experiment(seed=seed, n_repeats=n_repeats)


def step_midi(df, stim_cache):
    export_all(df, stim_cache, out_dir=MIDI_DIR)


def step_audio():
    convert_all(
        midi_root=MIDI_DIR,
        wav_root=WAV_DIR,
        mp3_root=MP3_DIR,
        soundfont=str(SOUNDFONT_PATH)
    )


def step_dataset(df):
    df = build_audio_map(df, mp3_root=MP3_DIR)

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(METADATA_PATH, index=False)

    return df


def step_analysis(mp3_dir):
    log("🧠 Running MFCC + UMAP analysis...")

    try:
        from analysis.pipelines.audio_space import run_audio_pipeline
    except Exception as e:
        log("❌ Analysis dependencies not available")
        print(e)
        return None

    return run_audio_pipeline(mp3_dir)


# =========================================================
# PIPELINE
# =========================================================

def run_pipeline(seed=42, n_repeats=8, skip_audio=False, run_analysis=False):

    ensure_data_dirs()

    total_steps = 4 if not skip_audio else 3
    pbar = tqdm(total=total_steps, desc="🚀 Pipeline", unit="step")

    # 1. GENERATE
    pbar.set_description("🎛️ Generating stimuli")
    df, stim_cache = step_generate(seed, n_repeats)
    pbar.update(1)

    # 2. MIDI
    pbar.set_description("🎼 Exporting MIDI")
    step_midi(df, stim_cache)
    pbar.update(1)

    # 3. AUDIO
    if not skip_audio:
        pbar.set_description("🎧 Rendering audio")
        step_audio()
        pbar.update(1)

    # 4. DATASET
    pbar.set_description("📊 Building dataset")
    df = step_dataset(df)
    pbar.update(1)

    pbar.close()

    # ANALYSIS
    analysis_result = None
    if run_analysis:
        analysis_result = step_analysis(MP3_DIR)
        log("📊 Embedding ready")

    log(f"✔ DONE → {len(df)} stimuli")

    return df, analysis_result


# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Groove Experiment Pipeline")

    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--analysis", action="store_true")

    args = parser.parse_args()

    if args.clean:
        clean()
        exit(0)

    df, analysis_result = run_pipeline(
        seed=args.seed,
        n_repeats=args.repeats,
        skip_audio=args.skip_audio,
        run_analysis=args.analysis
    )

    print("\n🔥 READY:", len(df), "stimuli")