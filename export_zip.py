from pathlib import Path
import zipfile
import fnmatch

# ==========================================
# CONFIGURATION
# ==========================================

# Dossier du projet à zipper
PROJECT_DIR = Path(".").resolve()

# Nom du zip généré
OUTPUT_ZIP = PROJECT_DIR / "claude_project_export.zip"

# Fichiers / dossiers à ignorer
EXCLUDE_PATTERNS = [
    "*.pyc",
    "*.pyo",
    "*.log",
    "*.tmp",
    "*.cache",
    ".DS_Store",
    ".env",
    ".venv/*",
    "venv/*",
    "node_modules/*",
    "dist/*",
    "build/*",
    "__pycache__/*",
    ".git/*",
    ".idea/*",
    ".vscode/*",
    "*.zip",
    "*.mp3",
    "*.mid",
    "*.csv",
]


# ==========================================
# HELPERS
# ==========================================

def should_exclude(path: Path) -> bool:
    """Retourne True si le fichier doit être ignoré."""

    relative = str(path.relative_to(PROJECT_DIR))

    for pattern in EXCLUDE_PATTERNS:
        if fnmatch.fnmatch(relative, pattern):
            return True

    return False


# ==========================================
# ZIP CREATION
# ==========================================

def create_zip():
    files_added = 0

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zipf:

        for path in PROJECT_DIR.rglob("*"):

            if path.is_dir():
                continue

            if should_exclude(path):
                print(f"[IGNORED] {path}")
                continue

            arcname = path.relative_to(PROJECT_DIR)
            zipf.write(path, arcname)
            files_added += 1

            print(f"[ADDED] {arcname}")

    print("\n==================================")
    print(f"ZIP CREATED: {OUTPUT_ZIP}")
    print(f"FILES ADDED: {files_added}")
    print("==================================")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    create_zip()

