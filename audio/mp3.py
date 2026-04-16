import subprocess
import os
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import MIDI_DIR, WAV_DIR, MP3_DIR, SOUNDFONT_PATH

SAMPLE_RATE = 44100
N_WORKERS = os.cpu_count() or 4


# =========================================================
# CORE CONVERSION
# =========================================================

def midi_to_audio_task(args):
    """
    Wrapper pour multiprocessing
    """
    midi_path, wav_path, mp3_path, soundfont = args

    try:
        midi_to_audio(midi_path, wav_path, mp3_path, soundfont)
        return (midi_path.name, True, None)
    except Exception as e:
        return (midi_path.name, False, str(e))


def midi_to_audio(
    midi_path,
    wav_path,
    mp3_path=None,
    soundfont=SOUNDFONT_PATH,
    make_mp3=True,
    keep_wav=False
):
    midi_path = str(midi_path)
    wav_path = str(wav_path)

    # ✅ CHECK SOUNDFONT
    if not Path(soundfont).exists():
        raise FileNotFoundError(f"SoundFont not found: {soundfont}")

    # ===============================
    # 1. MIDI → WAV
    # ===============================
    result = subprocess.run(
        [
            "fluidsynth",
            "-ni",
            str(soundfont),
            midi_path,
            "-F",
            wav_path,
            "-r",
            str(SAMPLE_RATE),
            "-g", "1.5"
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Fluidsynth failed:\n{result.stderr}")

    if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1000:
        raise RuntimeError(f"Empty WAV: {wav_path}")

    # ===============================
    # 2. CLEAN WAV
    # ===============================
    cleaned_wav = wav_path.replace(".wav", "_clean.wav")

    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", wav_path,
            "-af",
            "loudnorm=I=-14:TP=-1.0:LRA=11,highpass=f=30,lowpass=f=12000",
            cleaned_wav
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg clean failed:\n{result.stderr}")

    # ===============================
    # 3. WAV → MP3
    # ===============================
    if make_mp3 and mp3_path is not None:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", cleaned_wav,
                "-codec:a", "libmp3lame",
                "-qscale:a", "2",
                mp3_path
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg MP3 failed:\n{result.stderr}")

        if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) < 1000:
            raise RuntimeError(f"Empty MP3: {mp3_path}")

    # ===============================
    # 4. CLEAN
    # ===============================
    if not keep_wav:
        for f in [wav_path, cleaned_wav]:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass


# =========================================================
# PARALLEL BATCH
# =========================================================

def convert_all(
    midi_root=MIDI_DIR,
    wav_root=WAV_DIR,
    mp3_root=MP3_DIR,
    soundfont=SOUNDFONT_PATH,
    n_workers=N_WORKERS
):

    midi_root = Path(midi_root)
    wav_root = Path(wav_root)
    mp3_root = Path(mp3_root)

    wav_root.mkdir(parents=True, exist_ok=True)
    mp3_root.mkdir(parents=True, exist_ok=True)

    midi_files = list(midi_root.rglob("*.mid"))

    if len(midi_files) == 0:
        print("⚠️ No MIDI files found")
        return

    # ===============================
    # BUILD TASK LIST
    # ===============================
    tasks = []

    for midi_path in midi_files:
        rel = midi_path.relative_to(midi_root)

        wav_path = wav_root / rel.with_suffix(".wav")
        mp3_path = mp3_root / rel.with_suffix(".mp3")

        wav_path.parent.mkdir(parents=True, exist_ok=True)
        mp3_path.parent.mkdir(parents=True, exist_ok=True)

        tasks.append((midi_path, wav_path, mp3_path, soundfont))

    # ===============================
    # PARALLEL EXECUTION
    # ===============================
    print(f"\n⚡ Converting {len(tasks)} files with {n_workers} workers...\n")

    errors = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(midi_to_audio_task, t) for t in tasks]

        for f in tqdm(as_completed(futures), total=len(futures), desc="🎧 Audio", unit="file"):
            name, success, err = f.result()

            if not success:
                errors.append((name, err))
                tqdm.write(f"❌ {name}")

    # ===============================
    # REPORT
    # ===============================
    if errors:
        print(f"\n⚠️ {len(errors)} errors:")
        for name, err in errors[:5]:
            print(f"- {name}: {err}")
    else:
        print("\n✅ All files converted successfully")


# =========================================================
# DATASET
# =========================================================

def build_audio_map(df, mp3_root=MP3_DIR):

    mp3_root = Path(mp3_root)

    id_to_path = {}

    for p in mp3_root.rglob("*.mp3"):
        stem = p.stem
        idx = int(stem.split("_")[1])
        id_to_path[idx] = str(p)

    df = df.copy()
    df["mp3_path"] = df["id"].map(id_to_path)

    return df