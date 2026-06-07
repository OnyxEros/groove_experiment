"""
cli.py
======
Point d'entrée unique du système Groove Experiment.

v2 — Améliorations campagne :
    - cmd_generate : bloqué si .groove_locked existe (sauf --force)
    - cmd_lock     : crée/vérifie le verrou avec checksum MD5 des MP3
    - cmd_clean    : bloqué si .groove_locked existe
    - cmd_sync     : validation croisée stim_ids Supabase ↔ metadata.csv
    - cmd_status   : affiche couverture des stimuli (n réponses par stimulus)
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.panel import Panel
    from rich.columns import Columns
    from rich import box as rich_box
    _RICH = True
except ImportError:
    _RICH = False

from config import (
    ANALYSIS_DIR, METADATA_PATH, MIDI_DIR, MP3_DIR,
    PREVIEW_DIR, RESP_FILE, SOUNDFONT_PATH, WAV_DIR,
    BASE_DIR, SEED,
    ensure_data_dirs,
)

_console = Console() if _RICH else None

# ── Fichier de verrou campagne ────────────────────────────
LOCK_FILE = BASE_DIR / ".groove_locked"


# =========================================================
# AFFICHAGE
# =========================================================

def _print(msg: str, style: str = "bold green") -> None:
    if _RICH:
        _console.print(f"\n{msg}\n", style=style)
    else:
        print(f"\n{msg}\n")


def _warn(msg: str) -> None:
    if _RICH:
        _console.print(f"⚠️  {msg}", style="yellow")
    else:
        print(f"⚠️  {msg}", file=sys.stderr)


def _error(msg: str) -> None:
    if _RICH:
        _console.print(f"\n❌  {msg}\n", style="bold red")
    else:
        print(f"\n❌  {msg}\n", file=sys.stderr)


def _info(msg: str) -> None:
    if _RICH:
        _console.print(f"   {msg}", style="dim")
    else:
        print(f"   {msg}")


def _done() -> None:
    if _RICH:
        _console.print("\n[bold green]🔥  DONE[/bold green]\n")
    else:
        print("\n🔥  DONE\n")


def safe_exit(msg: str, code: int = 1) -> None:
    _error(msg)
    sys.exit(code)


@contextmanager
def step(label: str, dry_run: bool = False):
    if dry_run:
        if _RICH:
            _console.print(f"  [dim]DRY-RUN[/dim] [cyan]{label}[/cyan]")
        else:
            print(f"  [DRY-RUN] {label}")
        yield
        return

    start = time.perf_counter()

    if _RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold cyan]{label}[/bold cyan]"),
            TimeElapsedColumn(),
            console=_console,
            transient=True,
        ) as progress:
            progress.add_task("", total=None)
            try:
                yield
            except Exception:
                elapsed = time.perf_counter() - start
                _console.print(f"  ❌ [red]{label}[/red] failed after {elapsed:.1f}s")
                _console.print_exception(show_locals=False)
                raise
    else:
        print(f"\n▶  {label}…")
        try:
            yield
        except Exception as exc:
            elapsed = time.perf_counter() - start
            print(f"  FAILED after {elapsed:.1f}s: {exc}")
            traceback.print_exc()
            raise

    elapsed = time.perf_counter() - start
    if _RICH:
        _console.print(f"  ✔  [green]{label}[/green] [dim]({elapsed:.1f}s)[/dim]")
    else:
        print(f"  ✔  {label} ({elapsed:.1f}s)")


# =========================================================
# PREFLIGHT
# =========================================================

def _check_soundfont() -> bool:
    ok = Path(SOUNDFONT_PATH).exists()
    if not ok:
        _warn(f"SoundFont not found: {SOUNDFONT_PATH}")
    return ok


def _check_deps() -> dict[str, bool]:
    result: dict[str, bool] = {}
    for pkg in ["numpy", "pandas", "scipy", "sklearn", "umap", "shap", "statsmodels"]:
        try:
            __import__(pkg.replace("-", "_"))
            result[pkg] = True
        except ImportError:
            result[pkg] = False
    return result


# =========================================================
# VERROU CAMPAGNE
# =========================================================

def _check_lock(force: bool = False) -> None:
    """Bloque l'opération si le verrou campagne est actif."""
    if LOCK_FILE.exists() and not force:
        msg = LOCK_FILE.read_text().strip()
        safe_exit(
            f"Opération bloquée — campagne en cours.\n"
            f"  {LOCK_FILE.name} :\n"
            + "\n".join(f"    {line}" for line in msg.splitlines()) +
            f"\n\n  Pour forcer (DANGER) : ajouter --force"
        )


def _compute_mp3_checksum() -> tuple[str, int]:
    """Calcule le MD5 de l'ensemble des MP3 dans MP3_DIR."""
    mp3_files = sorted(MP3_DIR.rglob("*.mp3"))
    digest    = hashlib.md5()
    for f in mp3_files:
        digest.update(f.name.encode())   # nom dans le hash pour détecter les renommages
        digest.update(f.read_bytes())
    return digest.hexdigest(), len(mp3_files)


def cmd_lock(dry_run: bool = False) -> None:
    """
    Crée le fichier de verrou .groove_locked avec :
        - date de verrouillage
        - seed utilisée pour la génération
        - nombre de stimuli
        - MD5 de l'ensemble des MP3

    Si le verrou existe déjà, vérifie l'intégrité des MP3.
    """
    if dry_run:
        _print("🔒  [DRY-RUN] Verrouillage ignoré")
        return

    if not MP3_DIR.exists() or not list(MP3_DIR.rglob("*.mp3")):
        safe_exit(f"Aucun MP3 trouvé dans {MP3_DIR} — génère d'abord les stimuli.")

    import pandas as pd
    n_stimuli = len(pd.read_csv(METADATA_PATH)) if METADATA_PATH.exists() else "?"

    with step("Calcul du checksum MD5 des MP3"):
        checksum, n_mp3 = _compute_mp3_checksum()

    if LOCK_FILE.exists():
        # Mode vérification
        existing = LOCK_FILE.read_text()
        if f"md5={checksum}" in existing:
            _print("✔  Intégrité vérifiée — les MP3 n'ont pas changé")
        else:
            _warn(
                "⚠️  CHECKSUM DIVERGE — les MP3 ont été modifiés depuis le verrouillage !\n"
                "   Compare les MD5 dans .groove_locked avec le checksum actuel :"
            )
            _info(f"  MD5 actuel  : {checksum}")
            for line in existing.splitlines():
                if line.startswith("md5="):
                    _info(f"  MD5 verrou  : {line}")
        return

    # Création du verrou
    content = (
        f"date={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"seed={SEED}\n"
        f"n_stimuli={n_stimuli}\n"
        f"n_mp3={n_mp3}\n"
        f"md5={checksum}\n"
        f"note=Campagne en cours. Ne pas régénérer les stimuli.\n"
    )
    LOCK_FILE.write_text(content)
    _print(
        f"🔒  Verrou créé → {LOCK_FILE.name}\n"
        f"   {n_mp3} MP3 checksummés (MD5={checksum[:12]}…)"
    )
    _info("Pour vérifier l'intégrité plus tard : python cli.py --lock")
    _info("Pour déverrouiller manuellement    : rm .groove_locked")


# =========================================================
# STATUS
# =========================================================

def cmd_status() -> None:
    if not _RICH:
        print("Status requires `rich`.  Install with: pip install rich")
        return

    table = Table(
        title="🎧  Groove Experiment — État du système",
        show_header=True,
        header_style="bold magenta",
        box=rich_box.SIMPLE_HEAVY,
    )
    table.add_column("Composant",  style="cyan",  min_width=18)
    table.add_column("Chemin",     style="dim",   min_width=36)
    table.add_column("État",       justify="center")

    # ── Verrou ───────────────────────────────────────────
    if LOCK_FILE.exists():
        lock_info = LOCK_FILE.read_text().strip().replace("\n", " | ")
        table.add_row("🔒 Verrou campagne", str(LOCK_FILE), f"[yellow]ACTIF[/yellow]  {lock_info[:60]}…")
    else:
        table.add_row("🔒 Verrou campagne", str(LOCK_FILE), "[dim]—  (pas de verrou)[/dim]")

    dirs = {
        "MIDI dir":     MIDI_DIR,
        "MP3 dir":      MP3_DIR,
        "Preview dir":  PREVIEW_DIR,
        "Analysis dir": ANALYSIS_DIR,
        "Metadata":     METADATA_PATH,
        "SoundFont":    Path(SOUNDFONT_PATH),
    }
    for name, path in dirs.items():
        exists = path.exists()
        badge  = "[green]✔[/green]" if exists else "[red]✗[/red]"
        extra  = ""
        if exists and path.is_dir():
            n     = sum(1 for _ in path.rglob("*") if _.is_file())
            extra = f" ({n} fichiers)"
        elif exists and path.is_file():
            size  = path.stat().st_size
            extra = f" ({size/1024:.0f} KB)"
        table.add_row(name, str(path) + extra, badge)

    # ── Réponses locales + couverture stimuli ─────────────
    resp = Path(RESP_FILE)
    if resp.exists():
        try:
            import pandas as pd
            df = pd.read_csv(resp)
            n_resp  = len(df)
            n_parts = df["participant_id"].nunique() if "participant_id" in df.columns else "?"
            n_stims = df["stim_id"].nunique() if "stim_id" in df.columns else "?"

            # Couverture
            coverage_str = ""
            if METADATA_PATH.exists():
                meta       = pd.read_csv(METADATA_PATH)
                n_total    = len(meta)
                counts     = df.groupby("stim_id").size() if "stim_id" in df.columns else pd.Series(dtype=int)
                n_min2     = int((counts >= 2).sum())
                pct        = round(n_stims / n_total * 100) if n_total > 0 else 0
                pct_min2   = round(n_min2 / n_total * 100)  if n_total > 0 else 0
                coverage_str = (
                    f"  |  {n_stims}/{n_total} stimuli ({pct}%)"
                    f"  |  {n_min2} avec ≥2 rép. ({pct_min2}%)"
                )

            table.add_row(
                "responses.csv",
                str(resp),
                f"[green]✔  ({n_resp} rép. · {n_parts} part.{coverage_str})[/green]",
            )
        except Exception:
            table.add_row("responses.csv", str(resp), "[yellow]⚠  illisible[/yellow]")
    else:
        table.add_row("responses.csv", str(resp), "[dim]–  (pas encore synchronisé)[/dim]")

    # ── Run courant ───────────────────────────────────────
    current_run_file = Path(".current_run")
    if current_run_file.exists():
        run_path = Path(current_run_file.read_text().strip())
        if run_path.exists():
            figs = list(run_path.rglob("*.png")) + list(run_path.rglob("*.pdf"))
            table.add_row("Run courant", str(run_path.name), f"[green]✔  ({len(figs)} figures)[/green]")
        else:
            table.add_row("Run courant", str(run_path), "[red]✗  introuvable[/red]")

    table.add_section()
    for pkg, ok in _check_deps().items():
        badge = "[green]✔[/green]" if ok else "[yellow]–  (optionnel)[/yellow]"
        table.add_row(pkg, f"import {pkg}", badge)

    _console.print(table)


# =========================================================
# CLEAN
# =========================================================

def cmd_clean(targets: list[str], dry_run: bool = False, force: bool = False) -> None:
    if not targets:
        targets = ["all"]

    _print(
        f"🧹  Nettoyage : {targets}" + (" [DRY-RUN]" if dry_run else ""),
        style="yellow",
    )

    dispatch = {
        "outputs":   _clean_outputs,
        "metadata":  _clean_metadata,
        "responses": _clean_responses,
        "analysis":  _clean_analysis,
        "cache":     _clean_pycache,
    }

    if "all" in targets:
        targets = list(dispatch.keys())

    # Guard : "outputs" ou "all" = destruction potentielle des MP3
    if any(t in ("outputs", "all") for t in targets):
        _check_lock(force=force)

    for target in targets:
        if target not in dispatch:
            _warn(f"Cible inconnue : {target!r} — ignorée")
            continue
        if dry_run:
            _info(f"[DRY-RUN] supprimerait : {target}")
        else:
            dispatch[target]()

    _print("✔  Nettoyage terminé")


def _clean_outputs() -> None:
    for d in [MIDI_DIR, WAV_DIR, MP3_DIR, PREVIEW_DIR]:
        if d.exists():
            shutil.rmtree(d)
            _info(f"supprimé {d}")


def _clean_metadata() -> None:
    if METADATA_PATH.exists():
        METADATA_PATH.unlink()
        _info(f"supprimé {METADATA_PATH}")


def _clean_responses() -> None:
    resp = Path(RESP_FILE)
    if resp.exists():
        resp.unlink()
        _info(f"supprimé {resp}")
    else:
        _warn("Pas de cache de réponses local à supprimer.")


def _clean_analysis(subdirs: list[str] | None = None) -> None:
    if not ANALYSIS_DIR.exists():
        return
    if subdirs is None:
        shutil.rmtree(ANALYSIS_DIR)
        _info(f"supprimé {ANALYSIS_DIR}")
        from config import _CURRENT_RUN_FILE
        if _CURRENT_RUN_FILE.exists():
            _CURRENT_RUN_FILE.unlink()
            _info("supprimé .current_run")
    else:
        for sub in subdirs:
            p = ANALYSIS_DIR / sub
            if p.exists():
                shutil.rmtree(p)


def _clean_pycache() -> None:
    removed = 0
    for d in Path(".").rglob("__pycache__"):
        try:
            shutil.rmtree(d)
            removed += 1
        except Exception as exc:
            _warn(f"Impossible de supprimer {d}: {exc}")
    for f in Path(".").rglob("*.pyc"):
        try:
            f.unlink()
            removed += 1
        except Exception:
            pass
    _info(f"{removed} entrées cache supprimées")


# =========================================================
# §4.1 — GÉNÉRATION
# =========================================================

def cmd_generate(
    seed: int,
    n_repeats: int | None,
    skip_audio: bool = False,
    dry_run: bool = False,
    force: bool = False,
) -> None:
    """
    §4.1  Génère les stimuli, exporte les MIDI, rend les MP3.

    Bloqué si .groove_locked existe (campagne en cours).
    Utiliser --force pour outrepasser (DANGER).
    """
    # ── Guard verrou ──────────────────────────────────────
    _check_lock(force=force)

    if dry_run:
        _print("🎛️   [DRY-RUN] Pipeline de génération — rien ne sera écrit")
        return

    from groove.generator import run_experiment
    from audio.midi_export import export_all
    from audio.mp3 import convert_all, build_audio_map

    ensure_data_dirs()

    with step("Générer les stimuli  [groove.generator]"):
        df, stim_cache = run_experiment(seed=seed, n_repeats=n_repeats)
        _info(f"{len(df)} stimuli générés  (seed={seed})")

    with step("Exporter en MIDI  [audio.midi_export]"):
        export_all(df, stim_cache, out_dir=MIDI_DIR)

    if skip_audio:
        _warn("Rendu audio ignoré (--skip-audio)")
    else:
        if not _check_soundfont():
            safe_exit(
                f"SoundFont introuvable : {SOUNDFONT_PATH}\n"
                "Utilisez --skip-audio pour contourner."
            )
        with step("Rendre l'audio  WAV → MP3 (EQ + EBU R128)"):
            convert_all(
                midi_root=MIDI_DIR,
                wav_root=WAV_DIR,
                mp3_root=MP3_DIR,
                soundfont=str(SOUNDFONT_PATH),
            )

    with step("Construire metadata.csv  [build_audio_map]"):
        df = build_audio_map(df, mp3_root=MP3_DIR)
        METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(METADATA_PATH, index=False)

    n_mp3 = len(list(MP3_DIR.rglob("*.mp3"))) if MP3_DIR.exists() else 0
    _print(
        f"✔  Génération terminée — {len(df)} stimuli · {n_mp3} MP3 → {METADATA_PATH}"
    )
    _info("N'oublie pas de créer le verrou avant de lancer la campagne : python cli.py --lock")


# =========================================================
# §4.1.4 — VALIDATION STRUCTURELLE DU GÉNÉRATEUR
# =========================================================

def cmd_validate(dry_run: bool = False) -> None:
    if dry_run:
        _print("🔬  [DRY-RUN] Validation structurelle ignorée")
        return

    from analysis.viz.generative_validation import GenerativeValidation
    from analysis.viz.dataset_structure import DatasetStructureFigure
    from analysis.dataset.loader import load_dataset
    from config import get_current_run

    with step("Charger le corpus  [metadata.csv]"):
        df = load_dataset()
        _info(f"{len(df)} stimuli chargés")

    run_dir = get_current_run()
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    with step("Validation générative  (Panel A–D)"):
        GenerativeValidation().plot(
            df=df,
            path=fig_dir / "generative_validation.pdf",
            verbose=True,
        )

    with step("Structure du dataset  (Panel A–D)"):
        DatasetStructureFigure().plot(
            df=df,
            path=fig_dir / "dataset_structure.pdf",
            verbose=True,
        )

    _print(f"✔  Validation terminée → {fig_dir}")


# =========================================================
# §4.1 — PREVIEW
# =========================================================

def cmd_preview(seed: int = 42, dry_run: bool = False) -> None:
    if dry_run:
        _print("🎧  [DRY-RUN] Preview ignoré")
        return

    import numpy as np
    import pandas as pd
    from groove.generator import Grid, MicroTiming, Stimulus, Voices, _derive_seed
    from audio.midi_export import export_all
    from audio.mp3 import convert_all

    configs = [
        {"name": "baseline",   "phase": 0, "S_mv": 0, "D_mv": 1, "E_mv": 0.0, "P_mv": 0},
        {"name": "swing",      "phase": 0, "S_mv": 0, "D_mv": 1, "E_mv": 0.5, "P_mv": 0},
        {"name": "syncopated", "phase": 0, "S_mv": 2, "D_mv": 1, "E_mv": 0.0, "P_mv": 0},
    ]

    with step("Construire les stimuli de preview"):
        grid    = Grid()
        voices  = Voices(grid, seed=seed)
        rows, cache = [], {}
        for i, cfg in enumerate(configs):
            hihat_seed  = _derive_seed(seed, i, 0)
            timing_seed = _derive_seed(seed, i, 1)
            micro   = MicroTiming(
                rng=np.random.default_rng(timing_seed),
                step_duration=grid.step_duration,
            )
            builder = Stimulus(voices, micro)
            stim = builder.build(cfg, seed=hihat_seed)
            cache[i] = stim
            rows.append({"id": i, "stim_id": f"preview_{cfg['name']}", **cfg})

        df = pd.DataFrame(rows)
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    with step("Exporter MIDI preview"):
        export_all(df, cache, out_dir=PREVIEW_DIR)

    with step("Rendre audio preview"):
        convert_all(
            midi_root=PREVIEW_DIR,
            wav_root=PREVIEW_DIR / "wav",
            mp3_root=PREVIEW_DIR / "mp3",
            soundfont=str(SOUNDFONT_PATH),
        )

    _print(f"✔  Preview prêt → {PREVIEW_DIR / 'mp3'}")
    for cfg in configs:
        _info(f"  {cfg['name']:<12} S_mv={cfg['S_mv']} D_mv={cfg['D_mv']} E_mv={cfg['E_mv']}")


# =========================================================
# §4.2 — SERVEUR
# =========================================================

def cmd_serve(dry_run: bool = False) -> None:
    if dry_run:
        _print("🌐  [DRY-RUN] Serveur ignoré")
        return

    import uvicorn
    from config import PORT

    _print(f"🌐  Démarrage du serveur → http://localhost:{PORT}")
    uvicorn.run(
        "backend.api:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
        log_level="info",
    )


# =========================================================
# §4.2 — SYNC SUPABASE
# =========================================================

def cmd_sync(dry_run: bool = False) -> None:
    """
    §4.2  Rapatrie les réponses depuis Supabase vers le cache local.

    Valide en plus que les stim_ids dans Supabase correspondent à metadata.csv.
    """
    if dry_run:
        _print("☁️   [DRY-RUN] Sync Supabase ignoré")
        return

    import pandas as pd
    from infra.supabase_client import fetch_responses

    with step("Rapatrier les réponses ← Supabase"):
        data = fetch_responses()
        if not data:
            safe_exit("Aucune réponse dans Supabase (table 'responses' vide).")
        df = pd.DataFrame(data)

    # ── Validation croisée stim_ids ───────────────────────
    if METADATA_PATH.exists():
        with step("Valider les stim_ids ↔ metadata.csv"):
            meta     = pd.read_csv(METADATA_PATH)
            meta_ids = set(meta["stim_id"].astype(str)) if "stim_id" in meta.columns else set()
            resp_ids = set(df["stim_id"].astype(str))   if "stim_id" in df.columns  else set()

            orphans = resp_ids - meta_ids
            if orphans:
                _warn(
                    f"{len(orphans)} stim_id dans Supabase absents de metadata.csv :\n"
                    + "  " + ", ".join(sorted(orphans)[:10])
                    + ("…" if len(orphans) > 10 else "")
                    + "\n  → Ces réponses seront exclues de la jointure metadata × ratings."
                )
            else:
                _info(f"✔  Tous les stim_ids correspondent ({len(resp_ids)} IDs validés)")

            # Stimuli sans aucune réponse
            silent = meta_ids - resp_ids
            if silent:
                _warn(
                    f"{len(silent)} stimuli sans aucune réponse dans Supabase.\n"
                    + "  " + ", ".join(sorted(silent)[:10])
                    + ("…" if len(silent) > 10 else "")
                )

    with step("Écrire le cache local"):
        cache_path = Path(RESP_FILE)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)

    n_parts = df["participant_id"].nunique() if "participant_id" in df.columns else "?"
    n_stims = df["stim_id"].nunique()        if "stim_id" in df.columns        else "?"

    # Couverture rapide
    if METADATA_PATH.exists():
        n_total  = len(pd.read_csv(METADATA_PATH))
        counts   = df.groupby("stim_id").size() if "stim_id" in df.columns else pd.Series(dtype=int)
        n_min2   = int((counts >= 2).sum())
        pct_cov  = round(n_stims / n_total * 100) if n_total > 0 else 0
        pct_min2 = round(n_min2  / n_total * 100) if n_total > 0 else 0
        _print(
            f"✔  {len(df)} réponses  ·  {n_parts} participants\n"
            f"   Stimuli : {n_stims}/{n_total} couverts ({pct_cov}%)"
            f"  ·  {n_min2} avec ≥2 réponses ({pct_min2}%)"
        )
    else:
        _print(
            f"✔  {len(df)} réponses  ·  {n_parts} participants"
            f"  ·  {n_stims} stimuli  →  {RESP_FILE}"
        )


# =========================================================
# §5 — RUN + ANALYSIS
# =========================================================

def cmd_new_run() -> None:
    from config import new_run
    run_dir = new_run()
    _print(f"✔  Run courant : {run_dir}")


def cmd_analysis(
    mode: str = "groove",
    steps: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    if dry_run:
        _print(f"🧠  [DRY-RUN] Analysis — mode={mode}, steps={steps or 'default'}")
        return

    from analysis.core.run import run_analysis as _engine

    with step(f"Pipeline d'analyse  [mode={mode}]"):
        _engine(mode=mode, steps=steps, save=True, seed=42)


# =========================================================
# §6 — RÉGRESSION
# =========================================================

def cmd_regression(
    feature_set:    str  = "all",
    refresh:        bool = False,
    check_db:       bool = True,
    exclude_single: bool = True,
    dry_run:        bool = False,
) -> None:
    if dry_run:
        single_tag = "exclu" if exclude_single else "inclus"
        _print(f"📈  [DRY-RUN] Régression — features={feature_set}, n=1 {single_tag}")
        return

    from regression.run import run_regression

    single_tag = "exclu" if exclude_single else "inclus"
    with step(f"Régression  [features={feature_set} · n=1 {single_tag}]"):
        result = run_regression(
            feature_set=feature_set,
            refresh=refresh,
            check_db=check_db,
            exclude_single=exclude_single,
        )

    r2   = result.get("best_r2")
    best = result.get("best_model", "?")
    _print(
        f"✔  Régression terminée — meilleur={best}"
        + (f"  R²={r2:.3f}" if r2 is not None else "")
    )


def cmd_regression_all(
    refresh:        bool = False,
    check_db:       bool = True,
    exclude_single: bool = True,
    dry_run:        bool = False,
) -> None:
    for fs in ("design", "acoustic", "all"):
        cmd_regression(
            feature_set=fs,
            refresh=(refresh and fs == "design"),
            check_db=(check_db and fs == "design"),
            exclude_single=exclude_single,
            dry_run=dry_run,
        )


# =========================================================
# §6.3 — ALIGNEMENT PERCEPTIF
# =========================================================

def cmd_perception(refresh: bool = False, dry_run: bool = False) -> None:
    if dry_run:
        _print("🧠  [DRY-RUN] Alignement perceptif ignoré")
        return

    import pandas as pd
    from perception.alignment import fit_alignment, print_alignment_report
    from perception.loader import load_perceptual_dataset
    from perception.metrics import cluster_perception_diff

    with step("Charger le dataset joint  [agrégé par stimulus]"):
        meta = pd.read_csv(METADATA_PATH)
        if "stim_id" not in meta.columns and "id" in meta.columns:
            meta = meta.rename(columns={"id": "stim_id"})

        df = load_perceptual_dataset(embedding_df=meta, refresh=refresh)
        feature_cols = [c for c in ["D", "I", "V", "S", "E"] if c in df.columns]

        if not feature_cols:
            safe_exit(
                "Aucun descripteur émergent dans le dataset joint.\n"
                "Colonnes disponibles : " + str(list(df.columns)) + "\n"
                "Vérifiez que metadata.csv contient D, I, V, S, E."
            )

        if "groove_mean" not in df.columns:
            safe_exit("groove_mean absent du dataset joint.\nLancez d'abord : make sync")

    with step(f"Alignement Ridge  [features={feature_cols}]"):
        model, metrics = fit_alignment(
            df[feature_cols].values,
            df["groove_mean"].values,
        )
        print_alignment_report(metrics, label="+".join(feature_cols))

    r2 = metrics.get("r2_cv_mean", metrics.get("r2"))
    _print(f"📊  Alignement perceptif R² CV = {r2:.4f}")

    if "cluster" in df.columns:
        scores = cluster_perception_diff(
            df["cluster"].values,
            df["groove_mean"].values,
        )
        _print(f"📦  Groove moyen par cluster : {scores}")


# =========================================================
# §6.4 — ESPACE PERCEPTIF
# =========================================================

def cmd_perception_space(refresh: bool = False, dry_run: bool = False) -> None:
    if dry_run:
        _print("🧠  [DRY-RUN] Espace perceptif ignoré")
        return

    from perception_space.run import run_perception_space
    from perception.supabase_io import fetch_ratings

    with step("Charger les ratings  [format long, participant_id présent]"):
        df = fetch_ratings(refresh=refresh)
        df = df.rename(columns={"stim_id": "stimulus_id"})
        df["stimulus_id"] = df["stimulus_id"].astype(str)
        df = df.dropna(subset=["groove"])

        if "complexity" in df.columns and df["complexity"].isna().any():
            median_c = df["complexity"].median()
            df["complexity"] = df["complexity"].fillna(median_c)

        n_resp  = len(df)
        n_parts = df["participant_id"].nunique() if "participant_id" in df.columns else "?"
        n_stims = df["stimulus_id"].nunique()
        _info(f"{n_resp} réponses  ·  {n_parts} participants  ·  {n_stims} stimuli")

    with step("Géométrie de l'espace perceptif  [ICC · Mantel · k-NN]"):
        results = run_perception_space(perception_data=df)

    icc      = results.get("icc",      float("nan"))
    interp   = results.get("icc_interp", "?")
    mantel_r = results.get("mantel_r", float("nan"))
    mantel_p = results.get("mantel_p", float("nan"))
    status   = results.get("status", "?")
    fig_err  = results.get("fig_errors", [])

    if fig_err:
        _warn(f"Figures en échec ({len(fig_err)}) : {fig_err}")

    _print(
        f"✔  [{status}] ICC = {icc:.3f} ({interp})  ·  "
        f"Mantel r = {mantel_r:.3f}  p = {mantel_p:.4f}"
    )


# =========================================================
# FIGURES
# =========================================================

def cmd_figures(out: str = "figures_memoire", dry_run: bool = False) -> None:
    if dry_run:
        _print(f"🖼   [DRY-RUN] Figures ignorées → {out}/")
        return

    from config import get_current_run

    try:
        run_dir = get_current_run()
    except RuntimeError:
        safe_exit("Aucun run courant — lancez d'abord : make new-run && make analysis")

    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)

    collected = 0
    for ext in ("*.png", "*.pdf"):
        for src in run_dir.rglob(ext):
            parent   = src.parent.name
            dst_name = f"{parent}__{src.name}" if parent != run_dir.name else src.name
            shutil.copy2(src, dest / dst_name)
            collected += 1

    _print(f"✔  {collected} figures collectées → {dest}/")


# =========================================================
# PIPELINE COMPLET
# =========================================================

def cmd_thesis(
    refresh:        bool = False,
    exclude_single: bool = True,
    dry_run:        bool = False,
) -> None:
    if dry_run:
        _print("📖  [DRY-RUN] Pipeline thèse — séquence complète")
        for cmd in ["sync", "new-run", "analysis (mode=groove)", "regression-all",
                    "perception", "perc-space", "figures"]:
            _info(f"  → {cmd}")
        return

    _print("📖  Pipeline du mémoire", style="bold blue")

    cmd_sync(dry_run=False)
    cmd_new_run()
    cmd_analysis(mode="groove", dry_run=False)
    cmd_regression_all(refresh=False, check_db=False,
                       exclude_single=exclude_single, dry_run=False)
    cmd_perception(refresh=False, dry_run=False)
    cmd_perception_space(refresh=False, dry_run=False)
    cmd_figures(out="figures_memoire", dry_run=False)

    _print("📖  Pipeline du mémoire terminé", style="bold green")


# =========================================================
# DOCTOR
# =========================================================

def cmd_doctor() -> None:
    from perception.check_supabase import check_supabase
    from utils.env_check import run_env_check

    _print("🩺  Diagnostics…", style="cyan")
    ok_db  = check_supabase(refresh=False, verbose=True)
    ok_env = run_env_check()

    if ok_db and ok_env:
        _print("✔  Tous les diagnostics passés")
    else:
        _error("Certains diagnostics ont échoué — voir ci-dessus")
        sys.exit(1)


# =========================================================
# ARGUMENT PARSER
# =========================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="groove",
        description="🎧  Groove Experiment — CLI du mémoire",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
══════════════════════════════════════════════════════════
 PIPELINE DU MÉMOIRE (ordre recommandé)
══════════════════════════════════════════════════════════
 §4.1  Génération des stimuli
   python cli.py --generate                    → MIDI + MP3 + metadata.csv
   python cli.py --lock                        → verrou + checksum MD5 des MP3
   python cli.py --validate                    → figures validation §4.1.4
   python cli.py --preview                     → 3 stimuli de référence

 §4.2  Collecte des données
   python cli.py --serve                       → serveur expérience web
   python cli.py --sync                        → Supabase → responses.csv

 §5    Espace latent
   python cli.py --new-run                     → dossier run horodaté
   python cli.py --analysis                    → embeddings + UMAP + clusters

 §6    Modélisation statistique
   python cli.py --regression-all --refresh    → Ridge · EN · SVR · RF · LMM
   python cli.py --perception --refresh        → alignement Ridge
   python cli.py --perc-space                  → ICC · Mantel · géométrie

 Pipeline complet
   python cli.py --thesis                      → sync + analysis + modèles + figures

 Utilitaires
   python cli.py --status                      → état du système + couverture stimuli
   python cli.py --lock                        → créer/vérifier le verrou campagne
   python cli.py --doctor                      → diagnostic Supabase + env
   python cli.py --dry-run --thesis            → simulation sans écriture

══════════════════════════════════════════════════════════
 VERROU CAMPAGNE (.groove_locked)
══════════════════════════════════════════════════════════
   python cli.py --lock          → crée le verrou + checksum MD5 des MP3
   python cli.py --lock          → (si verrou existe) vérifie l'intégrité
   rm .groove_locked             → déverrouillage manuel
   python cli.py --generate --force  → forcer malgré le verrou (DANGER)
   python cli.py --clean --force     → forcer le nettoyage (DANGER)

══════════════════════════════════════════════════════════
""",
    )

    # ── §4.1 GÉNÉRATION ──────────────────────────────────────────────────────
    g = parser.add_argument_group("§4.1  Génération des stimuli")
    g.add_argument("--generate",   action="store_true",
                   help="Génère MIDI + MP3 + metadata.csv (bloqué si .groove_locked)")
    g.add_argument("--lock",       action="store_true",
                   help="Crée/vérifie le verrou campagne avec checksum MD5")
    g.add_argument("--validate",   action="store_true",
                   help="Figures de validation du générateur (§4.1.4)")
    g.add_argument("--preview",    action="store_true",
                   help="3 stimuli de référence pour écoute comparative")
    g.add_argument("--seed",       type=int, default=42,   metavar="N",
                   help="Graine aléatoire (défaut : 42)")
    g.add_argument("--repeats",    type=int, default=None, metavar="N",
                   help="Répétitions par condition (écrase config.py)")
    g.add_argument("--skip-audio", action="store_true",
                   help="Ignorer le rendu audio (MIDI seulement)")
    g.add_argument("--force",      action="store_true",
                   help="Ignore le verrou .groove_locked — DANGER")

    # ── §4.2 COLLECTE ────────────────────────────────────────────────────────
    g = parser.add_argument_group("§4.2  Collecte des données")
    g.add_argument("--serve",  action="store_true",
                   help="Démarre le serveur FastAPI de l'expérience")
    g.add_argument("--sync",   action="store_true",
                   help="Supabase → responses.csv + validation stim_ids")

    # ── §5 ANALYSE ───────────────────────────────────────────────────────────
    g = parser.add_argument_group("§5  Espace latent")
    g.add_argument("--new-run",    action="store_true",
                   help="Crée un dossier de run horodaté")
    g.add_argument("--analysis",   action="store_true",
                   help="Pipeline embeddings + UMAP + clustering + export")
    g.add_argument("--analysis-only", action="store_true",
                   help="Alias de --analysis (rétro-compat)")
    g.add_argument(
        "--analysis-mode",
        default="groove",
        choices=["full", "groove", "full_with_clustering", "audio"],
        metavar="MODE",
        help="Mode du pipeline d'analyse (défaut : groove)",
    )
    g.add_argument("--steps", nargs="+", metavar="STEP",
                   help="Steps manuels (écrase --analysis-mode)")

    # ── §6 MODÉLISATION ──────────────────────────────────────────────────────
    g = parser.add_argument_group("§6  Modélisation statistique")
    g.add_argument("--regression",     action="store_true",
                   help="Régression groove (1 feature set)")
    g.add_argument("--regression-all", action="store_true",
                   help="Régression groove (design + acoustic + all)")
    g.add_argument(
        "--feature-set",
        default="all",
        choices=["design", "acoustic", "all", "interactions", "predictability"],
        metavar="FS",
        help="Feature set pour --regression (défaut : all)",
    )
    g.add_argument(
        "--include-single",
        action="store_true",
        default=False,
        help="Inclut les stimuli n=1 dans la régression",
    )
    g.add_argument("--perception",       action="store_true",
                   help="Alignement Ridge espace latent → groove_mean")
    g.add_argument("--perception-space", action="store_true",
                   help="ICC · Test de Mantel · géométrie locale k-NN")
    g.add_argument("--no-check-db",      action="store_true",
                   help="Ignore la vérification Supabase avant régression")

    # ── PIPELINE THÈSE ───────────────────────────────────────────────────────
    g = parser.add_argument_group("Pipeline complet")
    g.add_argument("--thesis",      action="store_true",
                   help="Pipeline thèse complet")
    g.add_argument("--figures",     action="store_true",
                   help="Collecte toutes les figures → figures_memoire/")
    g.add_argument("--figures-out", default="figures_memoire", metavar="DIR",
                   help="Dossier destination pour --figures")

    # ── INFRA ────────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Infrastructure")
    g.add_argument("--refresh", action="store_true",
                   help="Force le re-fetch Supabase (ignore le cache local)")

    # ── MAINTENANCE ──────────────────────────────────────────────────────────
    g = parser.add_argument_group("Maintenance")
    g.add_argument(
        "--clean",
        nargs="*",
        choices=["all", "outputs", "metadata", "responses", "analysis", "cache"],
        metavar="TARGET",
        help="Nettoie les artefacts (bloqué par .groove_locked sauf --force)",
    )
    g.add_argument("--status",  action="store_true",
                   help="État du système + couverture stimuli")
    g.add_argument("--doctor",  action="store_true",
                   help="Diagnostic Supabase + environnement")
    g.add_argument("--dry-run", action="store_true",
                   help="Simulation — rien n'est écrit")

    return parser


_ACTION_FLAGS = {
    "generate", "lock", "validate", "preview",
    "serve", "sync",
    "new_run", "analysis", "analysis_only",
    "regression", "regression_all",
    "perception", "perception_space",
    "thesis", "figures",
}


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    dry    = args.dry_run
    force  = args.force

    exclude_single = not args.include_single

    if dry:
        _print("🔍  DRY-RUN — rien ne sera écrit ni exécuté", style="bold yellow")

    # ── Commandes sans pipeline ──────────────────────────────────────────────
    if args.status:
        cmd_status()
        return

    if args.doctor:
        cmd_doctor()
        return

    if args.clean is not None:
        cmd_clean(args.clean or ["all"], dry_run=dry, force=force)
        return

    if not any(getattr(args, f, False) for f in _ACTION_FLAGS):
        parser.print_help()
        return

    # ── §4.1 Génération ──────────────────────────────────────────────────────
    if args.generate:
        ensure_data_dirs()
        cmd_generate(
            seed=args.seed,
            n_repeats=args.repeats,
            skip_audio=args.skip_audio,
            force=force,
            dry_run=dry,
        )

    if args.lock:
        cmd_lock(dry_run=dry)

    if args.validate:
        cmd_validate(dry_run=dry)

    if args.preview:
        ensure_data_dirs()
        cmd_preview(seed=args.seed, dry_run=dry)

    # ── §4.2 Collecte ────────────────────────────────────────────────────────
    if args.serve:
        cmd_serve(dry_run=dry)
        return

    if args.sync:
        cmd_sync(dry_run=dry)

    # ── §5 Analyse ───────────────────────────────────────────────────────────
    if args.new_run:
        cmd_new_run()

    if args.analysis or args.analysis_only:
        cmd_analysis(mode=args.analysis_mode, steps=args.steps, dry_run=dry)

    # ── §6 Modélisation ──────────────────────────────────────────────────────
    if args.regression:
        cmd_regression(
            feature_set=args.feature_set,
            refresh=args.refresh,
            check_db=not args.no_check_db,
            exclude_single=exclude_single,
            dry_run=dry,
        )

    if args.regression_all:
        cmd_regression_all(
            refresh=args.refresh,
            check_db=not args.no_check_db,
            exclude_single=exclude_single,
            dry_run=dry,
        )

    if args.perception:
        cmd_perception(refresh=args.refresh, dry_run=dry)

    if args.perception_space:
        cmd_perception_space(refresh=args.refresh, dry_run=dry)

    # ── Pipeline thèse ───────────────────────────────────────────────────────
    if args.thesis:
        cmd_thesis(
            refresh=args.refresh,
            exclude_single=exclude_single,
            dry_run=dry,
        )

    if args.figures:
        cmd_figures(out=args.figures_out, dry_run=dry)

    _done()


if __name__ == "__main__":
    main()