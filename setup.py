import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SOUNDFONT_PATH = BASE_DIR / "soundfont.sf2"

def run(cmd):
    print(f"\n👉 {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def check_command(cmd):
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, 
stderr=subprocess.DEVNULL)
        return True
    except:
        return False


def install_soundfont():
    if SOUNDFONT_PATH.exists():
        print("✔ SoundFont already present")
        return

    print("⬇️ Downloading SoundFont...")
    run(
        "curl -L 
https://github.com/urish/cinto/raw/master/media/FluidR3_GM.sf2 -o 
soundfont.sf2"
    )
    print("✔ SoundFont installed")


def check_audio_tools():
    print("\n🔍 Checking audio dependencies...")

    for tool in ["ffmpeg", "fluidsynth"]:
        result = subprocess.run(
            ["which", tool],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        if result.returncode != 0:
            print(f"❌ {tool} not found")
            print("👉 Install with conda:")
            print(f"   conda install -c conda-forge {tool}")
            sys.exit(1)
        else:
            print(f"✔ {tool} OK")


def install_python_deps():
    print("\n📦 Installing Python dependencies...")
    run("pip install -r requirements.txt")


def main():
    print("\n🚀 Groove Experiment Setup\n")

    install_python_deps()
    check_audio_tools()
    install_soundfont()

    print("\n🔥 Setup complete! You can now run:")
    print("   make run_all")


if __name__ == "__main__":
    main()
