"""
perception/check_supabase.py
============================
Diagnostic de la connexion Supabase et de la cohérence des données.

Usage direct :
    python perception/check_supabase.py

Usage depuis le code :
    from perception.check_supabase import check_supabase
    ok = check_supabase(refresh=False)
"""

import sys
from pathlib import Path

# Permet l'exécution directe depuis n'importe où
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from infra.supabase_client import get_supabase
from perception.supabase_io import fetch_ratings


# =========================================================
# DIAGNOSTIC PRINCIPAL
# =========================================================

def check_supabase(refresh: bool = False, verbose: bool = True) -> bool:
    """
    Vérifie la connexion Supabase et la cohérence des données.

    Args:
        refresh: re-fetch les réponses même si le cache local existe.
        verbose: affiche le rapport dans le terminal.

    Returns:
        True si tout est OK, False si un problème bloquant est détecté.
    """
    _sep()
    _log("DIAGNOSTIC SUPABASE")
    _sep()

    ok = True

    # ── 1. Variables d'environnement ─────────────────────
    _section("1/4  Variables d'environnement")

    if not config.SUPABASE_URL:
        _fail("SUPABASE_URL manquante dans .env / variables Render")
        return False
    if not config.SUPABASE_KEY:
        _fail("SUPABASE_KEY manquante dans .env / variables Render")
        return False

    _ok(f"SUPABASE_URL  : {config.SUPABASE_URL[:35]}…")
    _ok(f"SUPABASE_KEY  : {config.SUPABASE_KEY[:20]}…")

    # ── 2. Connexion client ───────────────────────────────
    _section("2/4  Connexion client")

    try:
        client = get_supabase()
        _ok("Client Supabase initialisé")
    except Exception as e:
        _fail(f"Échec initialisation : {e}")
        return False

    # ── 3. Table responses ────────────────────────────────
    _section("3/4  Table 'responses'")

    try:
        result = client.table("responses").select("*", count="exact").limit(1).execute()
        count  = result.count if hasattr(result, "count") and result.count else len(result.data)
        _ok(f"Table accessible — {count}+ ligne(s)")
    except Exception as e:
        _fail(f"Table 'responses' inaccessible : {e}")
        ok = False

    # ── 4. Chargement des réponses via le loader ──────────
    _section("4/4  Chargement des réponses")

    try:
        df = fetch_ratings(refresh=refresh)

        if df.empty:
            _warn("Aucune réponse valide après nettoyage")
            _hint("→ Lance la collecte via l'interface web puis relance avec --refresh")
            ok = False
        else:
            n_resp   = len(df)
            n_parts  = df["participant_id"].nunique() if "participant_id" in df.columns else "?"
            n_stims  = df["stim_id"].nunique()
            g_mean   = df["groove"].mean()
            g_std    = df["groove"].std()

            _ok(f"{n_resp} réponses  |  {n_parts} participants  |  {n_stims} stimuli évalués")
            _ok(f"groove : {g_mean:.2f} ± {g_std:.2f}  (échelle 1–7)")

            if "complexity" in df.columns:
                _ok(f"complexity : {df['complexity'].mean():.2f} ± {df['complexity'].std():.2f}")

            if n_stims < 5:
                _warn(f"Seulement {n_stims} stimuli évalués — la régression sera peu fiable")
                _hint("→ Collecte plus de données avant de lancer --regression")

            # Aperçu stim_id pour diagnostic jointure
            sample_ids = df["stim_id"].unique()[:4].tolist()
            _info(f"stim_id sample : {sample_ids}")

    except ValueError as e:
        _fail(str(e))
        ok = False
    except Exception as e:
        _fail(f"Erreur inattendue : {e}")
        import traceback
        traceback.print_exc()
        ok = False

    _sep()
    if ok:
        _log("✅  Tout est OK")
    else:
        _log("⚠️  Des problèmes ont été détectés — voir ci-dessus")
    _sep()

    return ok


# =========================================================
# HELPERS D'AFFICHAGE
# =========================================================

def _sep():   print("\n" + "═" * 55)
def _section(t): print(f"\n  [{t}]")
def _log(t):  print(f"  {t}")
def _ok(t):   print(f"    ✅  {t}")
def _warn(t): print(f"    ⚠️   {t}")
def _fail(t): print(f"    ❌  {t}")
def _hint(t): print(f"    💡  {t}")
def _info(t): print(f"    ℹ️   {t}")


# =========================================================
# POINT D'ENTRÉE DIRECT
# =========================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diagnostic Supabase — Groove Study")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-fetch les réponses depuis Supabase (ignore le cache local)")
    args = parser.parse_args()

    success = check_supabase(refresh=args.refresh)
    sys.exit(0 if success else 1)