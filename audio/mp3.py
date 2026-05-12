"""
audio/mp3.py
============
Conversion MIDI → WAV → MP3 avec post-traitement audio uniforme.

Post-traitement v2 :
    Une passe ffmpeg EQ + compression est appliquée de manière identique
    sur tous les stimuli avant l'encodage MP3. Étant strictement uniforme,
    ce traitement n'introduit aucun biais différentiel entre conditions —
    même logique que la normalisation de loudness standard en psychoacoustique.

    Objectif : corriger les déséquilibres spectraux typiques du rendu
    fluidsynth (kick trop sourd, hi-hat trop clinquant, pad qui noie
    les percussions) pour rendre les variations rythmiques plus saillantes
    perceptivement, notamment pour les participants sans culture musicale.

    Chaîne de traitement (dans l'ordre) :
        1. Boost sub-kick  (+3dB à 80Hz)  — masse et présence du kick
        2. Dévase          (-2dB à 200Hz) — retire la boue typique GM
        3. Présence mid    (+2dB à 2kHz)  — articulation des percussions
        4. Adoucit les hats(-3dB à 8kHz)  — hi-hat moins agressif
        5. Compresseur léger              — réduit les écarts de dynamique
        6. loudnorm (EBU R128)            — normalisation à -16 LUFS,
                                           niveau uniforme entre stimuli
"""

import subprocess
import os
import time
import shutil
import re
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import MIDI_DIR, WAV_DIR, MP3_DIR, SOUNDFONT_PATH


# =========================================================
# GLOBAL CONFIG
# =========================================================

SAMPLE_RATE = 44100
N_WORKERS   = min(4, os.cpu_count() or 1)

# ── Chaîne de filtres ffmpeg ──────────────────────────────
# Appliquée uniformément sur tous les stimuli.
# Chaque filtre est commenté avec sa justification perceptive.
_EQ_FILTER = ",".join([
    # 1. Boost sub-kick : +3dB centré sur 80Hz (largeur 2 octaves)
    #    → donne de la masse au kick fluidsynth, souvent trop sourd
    "equalizer=f=80:width_type=o:width=2:g=3",

    # 2. Dévase : -2dB centré sur 200Hz (largeur 1.5 octave)
    #    → retire l'accumulation basse-fréquence typique des soundfonts GM
    #      qui rend le rendu "boueux"
    "equalizer=f=200:width_type=o:width=1.5:g=-2",

    # 3. Présence mid : +2dB centré sur 2kHz (largeur 1.5 octave)
    #    → améliore l'articulation et l'attaque des percussions,
    #      rend les variations rythmiques plus distinctes
    "equalizer=f=2000:width_type=o:width=1.5:g=2",

    # 4. Adoucit les hauts : -3dB centré sur 8kHz (largeur 1.5 octave)
    #    → réduit l'agressivité du hi-hat fluidsynth qui peut fatiguer
    #      l'écoute et masquer les variations structurelles
    "equalizer=f=8000:width_type=o:width=1.5:g=-3",

    # 5. Compresseur léger
    #    threshold=-18dB, ratio=3:1, attack=5ms, release=80ms
    #    → réduit les pics dynamiques sans écraser le groove naturel,
    #      équilibre les voix entre elles (kick vs hi-hat vs pad)
    "acompressor=threshold=0.125:ratio=3:attack=5:release=80:makeup=1.5",

    # 6. Normalisation EBU R128 à -16 LUFS
    #    → niveau d'écoute uniforme entre tous les stimuli,
    #      élimine les différences de volume perçu qui pourraient
    #      biaiser les jugements de groove (effet loudness)
    "loudnorm=I=-16:LRA=7:TP=-1.5",
])


# =========================================================
# UTILS
# =========================================================

def check_binary(name: str):
    if shutil.which(name) is None:
        raise RuntimeError(f"{name} not found in PATH")


def safe_path(p: Path | str) -> str:
    return str(Path(p).resolve())


def wait_for_file_ready(path: str, timeout: float = 3.0) -> None:
    start     = time.time()
    last_size = -1
    while time.time() - start < timeout:
        if not os.path.exists(path):
            time.sleep(0.05)
            continue
        size = os.path.getsize(path)
        if size > 1000 and size == last_size:
            return
        last_size = size
        time.sleep(0.05)
    raise RuntimeError(f"File not ready: {path}")


# =========================================================
# CORE CONVERSION
# =========================================================

def midi_to_audio(
    midi_path,
    wav_path,
    mp3_path,
    soundfont=SOUNDFONT_PATH,
):
    midi_path = safe_path(midi_path)
    wav_path  = safe_path(wav_path)
    mp3_path  = safe_path(mp3_path)
    soundfont = safe_path(soundfont)

    check_binary("fluidsynth")
    check_binary("ffmpeg")

    if not Path(soundfont).exists():
        raise FileNotFoundError(f"SoundFont not found: {soundfont}")

    # ── 1. MIDI → WAV (fluidsynth) ─────────────────────────
    subprocess.run(
        [
            "fluidsynth",
            "-ni",
            soundfont,
            midi_path,
            "-F", wav_path,
            "-r", str(SAMPLE_RATE),
            "-g", "1.0",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    wait_for_file_ready(wav_path)

    # ── 2. WAV → MP3 avec EQ + compression uniforme ────────
    #
    #    La chaîne _EQ_FILTER est appliquée de manière identique
    #    sur tous les stimuli (paramètre global, non conditionnel).
    #    Aucune différence de traitement entre conditions.
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", wav_path,
            "-af", _EQ_FILTER,
            "-codec:a", "libmp3lame",
            "-qscale:a", "2",
            mp3_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    if not os.path.exists(mp3_path):
        raise RuntimeError(f"MP3 not created: {mp3_path}")


def midi_to_audio_task(args):
    midi_path, wav_path, mp3_path, soundfont = args
    try:
        midi_to_audio(midi_path, wav_path, mp3_path, soundfont)
        return (midi_path.name, True, None)
    except Exception as e:
        return (midi_path.name, False, str(e))


# =========================================================
# BATCH CONVERSION
# =========================================================

def convert_all(
    midi_root=MIDI_DIR,
    wav_root=WAV_DIR,
    mp3_root=MP3_DIR,
    soundfont=SOUNDFONT_PATH,
    n_workers=N_WORKERS,
):
    midi_root = Path(midi_root)
    wav_root  = Path(wav_root)
    mp3_root  = Path(mp3_root)

    wav_root.mkdir(parents=True, exist_ok=True)
    mp3_root.mkdir(parents=True, exist_ok=True)

    midi_files = list(midi_root.rglob("*.mid"))

    if not midi_files:
        print("⚠️ No MIDI files found")
        return

    tasks = []
    for midi_path in midi_files:
        rel      = midi_path.relative_to(midi_root)
        wav_path = wav_root / rel.with_suffix(".wav")
        mp3_path = mp3_root / rel.with_suffix(".mp3")
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        mp3_path.parent.mkdir(parents=True, exist_ok=True)
        tasks.append((midi_path, wav_path, mp3_path, soundfont))

    print(f"\n⚡ Converting {len(tasks)} files with {n_workers} workers…")
    print(f"   EQ + compression : activés (traitement uniforme)\n")

    errors = []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(midi_to_audio_task, t) for t in tasks]

        for f in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="🎧 Audio",
            unit="file",
        ):
            name, success, err = f.result()
            if not success:
                errors.append((name, err))
                tqdm.write(f"\n❌ {name}\n{err}\n")

    if errors:
        print(f"\n⚠️ {len(errors)} errors:")
        for name, err in errors[:5]:
            print(f"  - {name}: {err}")
    else:
        print("\n✅ All files converted successfully")

    # Nettoyage WAV intermédiaires
    if wav_root.exists():
        shutil.rmtree(wav_root)
    print(f"🧹 Removed WAV directory: {wav_root}")


# =========================================================
# DATASET
# =========================================================

def build_audio_map(df, mp3_root=MP3_DIR):
    mp3_root   = Path(mp3_root)
    id_to_path = {}

    for p in mp3_root.rglob("*.mp3"):
        match = re.search(r"stim_(\d+)", p.stem)
        if match:
            idx             = int(match.group(1))
            id_to_path[idx] = safe_path(p)

    df             = df.copy()
    df["mp3_path"] = df["id"].map(id_to_path)

    return df