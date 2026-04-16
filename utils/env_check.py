import shutil
import importlib
import sys
import subprocess

from config import SOUNDFONT_PATH


REQUIRED_BINARIES = [
    "fluidsynth",
    "ffmpeg",
]

REQUIRED_PYTHON_LIBS = [
    "numpy",
    "pandas",
    "pretty_midi",
]

OPTIONAL_LIBS = [
    "librosa",
    "plotly",
    "umap",
]


# =========================================================
# SYSTEM BINARIES
# =========================================================

def check_binaries():
    print("\n🔍 Checking system binaries...\n")

    missing = []

    for b in REQUIRED_BINARIES:
        path = shutil.which(b)

        if path is None:
            print(f"❌ Missing: {b}")
            missing.append(b)
        else:
            print(f"✅ Found: {b} → {path}")

    return missing


def check_fluid_synth_runtime():
    """
    Vérifie que fluidsynth fonctionne réellement
    """
    try:
        subprocess.run(
            ["fluidsynth", "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("✅ fluidsynth runtime OK")
        return True
    except Exception:
        print("❌ fluidsynth not working at runtime")
        return False


def check_soundfont():
    """
    CRITIQUE pour ton pipeline audio
    """
    print("\n🎼 Checking soundfont...\n")

    if SOUNDFONT_PATH.exists():
        print(f"✅ Soundfont found → {SOUNDFONT_PATH}")
        return True
    else:
        print(f"❌ Missing soundfont → {SOUNDFONT_PATH}")
        print("\n💡 Fix:")
        print("  place a .sf2 file here or update config.py")
        return False


# =========================================================
# PYTHON LIBS
# =========================================================

def check_python_libs():
    print("\n🐍 Checking Python libraries...\n")

    missing = []

    for lib in REQUIRED_PYTHON_LIBS:
        try:
            importlib.import_module(lib)
            print(f"✅ {lib}")
        except ImportError:
            print(f"❌ {lib}")
            missing.append(lib)

    return missing


def check_optional_libs():
    print("\n🧪 Optional libraries (analysis only)...\n")

    for lib in OPTIONAL_LIBS:
        try:
            importlib.import_module(lib)
            print(f"✅ {lib}")
        except ImportError:
            print(f"⚠️ {lib} (optional)")


# =========================================================
# MAIN CHECK
# =========================================================

def run_env_check(strict=True):
    print("\n==============================")
    print(" ENVIRONMENT CHECK 🚀")
    print("==============================")

    bin_missing = check_binaries()
    py_missing = check_python_libs()
    check_optional_libs()

    fluid_ok = check_fluid_synth_runtime()
    soundfont_ok = check_soundfont()

    errors = []

    if bin_missing:
        errors.extend(bin_missing)

    if py_missing:
        errors.extend(py_missing)

    if not fluid_ok:
        errors.append("fluidsynth runtime")

    if not soundfont_ok:
        errors.append("soundfont missing")

    # =====================================================
    # REPORT
    # =====================================================

    if errors:
        print("\n❌ Environment not ready\n")

        print("Issues detected:")
        for e in errors:
            print(f"  - {e}")

        if bin_missing:
            print("\n📦 Install system deps:")
            print("  brew install " + " ".join(bin_missing))

        if py_missing:
            print("\n🐍 Install python deps:")
            print("  pip install " + " ".join(py_missing))

        print("\n🎼 Audio requirement:")
        print(f"  Soundfont required at: {SOUNDFONT_PATH}")

        if strict:
            sys.exit(1)

    else:
        print("\n✅ Environment OK\n")