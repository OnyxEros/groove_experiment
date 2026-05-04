from pathlib import Path

def get_latest_run(base_dir="data/analysis") -> Path | None:
    base = Path(base_dir)
    runs = sorted(base.glob("run_*"))

    if not runs:
        return None

    return runs[-1]