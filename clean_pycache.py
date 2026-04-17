from pathlib import Path
import shutil

def clean_pycache(root="."):
    root = Path(root)

    removed_dirs = 0
    removed_files = 0

    # Supprimer __pycache__
    for pycache in root.rglob("__pycache__"):
        if pycache.is_dir():
            shutil.rmtree(pycache)
            removed_dirs += 1
            print(f"🗑️ removed dir: {pycache}")

    # Supprimer .pyc
    for pyc in root.rglob("*.pyc"):
        try:
            pyc.unlink()
            removed_files += 1
            print(f"🗑️ removed file: {pyc}")
        except Exception as e:
            print(f"⚠️ could not remove {pyc}: {e}")

    print("\n✔ Cleanup done")
    print(f"   - __pycache__ dirs removed: {removed_dirs}")
    print(f"   - .pyc files removed: {removed_files}")


if __name__ == "__main__":
    clean_pycache()
