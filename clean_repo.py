from pathlib import Path
import re

ROOT = Path("regression")

# lignes dangereuses : uniquement des traits Unicode
BROKEN_LINE = re.compile(r"^\s*[─-]{3,}\s*$")

def fix_file(path: Path):
    changed = False
    new_lines = []

    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:

        # uniquement lignes 100% composées de tirets unicode
        stripped = line.strip()

        if stripped and all(c == "─" for c in stripped):
            new_lines.append("# ─────────────────────────────\n")
            changed = True
            continue

        new_lines.append(line)

    if changed:
        with path.open("w", encoding="utf-8") as f:
            f.writelines(new_lines)

    return changed


def main():
    files = list(ROOT.rglob("*.py"))

    modified = []

    for f in files:
        try:
            if fix_file(f):
                modified.append(str(f))
        except Exception as e:
            print(f"[ERROR] {f}: {e}")

    print("\n──────── CLEAN REPORT ────────")
    print(f"Files scanned : {len(files)}")
    print(f"Files fixed   : {len(modified)}")

    if modified:
        print("\nFixed files:")
        for m in modified:
            print(" -", m)


if __name__ == "__main__":
    main()
