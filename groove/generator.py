"""
groove/generator.py
===================
Générateur de stimuli rythmiques pour l'expérience groove.

Architecture :
    Grid        — paramètres temporels (BPM, résolution)
    Voices      — patterns rythmiques (kick, snare, hihat)
    MicroTiming — déviations temporelles expressives
    Stimulus    — assemblage d'un stimulus complet
    Metrics     — métriques acoustiques extraites
    build_design  — design expérimental factoriel
    run_experiment — point d'entrée principal
"""

import numpy as np
import itertools
import pandas as pd

import config


# =========================================================
# GRID
# =========================================================

class Grid:
    def __init__(self):
        self.steps_per_bar  = config.STEPS_PER_BAR
        self.n_steps        = self.steps_per_bar * config.BARS
        self.bpm            = config.BPM
        self.step_duration  = config.step_duration_seconds()
        # AMÉLIORATION : utilise la fonction centralisée plutôt que
        # de recalculer 60 / (bpm * spb/4) en dur ici.


# =========================================================
# VOICES
# =========================================================

class Voices:
    def __init__(self, grid: Grid, seed=None):
        self.n_steps       = grid.n_steps
        self.steps_per_bar = grid.steps_per_bar
        self.rng           = np.random.default_rng(seed)

    # ── Helpers ──────────────────────────────────────────

    def empty(self) -> np.ndarray:
        return np.zeros(self.n_steps, dtype=np.float64)

    # ── Kick ─────────────────────────────────────────────

    def kick(self) -> np.ndarray:
        """Kick sur les temps 1 et 3 (downbeat + midbeat)."""
        p   = self.empty()
        bar = self.steps_per_bar
        for b in range(self.n_steps // bar):
            o       = b * bar
            p[o]            = 1   # temps 1
            p[o + bar // 2] = 1   # temps 3
        return p

    # ── Snare ────────────────────────────────────────────

    def snare(self) -> np.ndarray:
        """Snare sur les backbeats (temps 2 et 4)."""
        p   = self.empty()
        bar = self.steps_per_bar
        for b in range(self.n_steps // bar):
            o               = b * bar
            p[o + bar // 4]     = 1   # temps 2
            p[o + 3 * bar // 4] = 1   # temps 4
        return p

    # ── Hi-hat ───────────────────────────────────────────

    def hihat(
        self,
        sync_level:    int = 0,
        density_level: int = 1,
        seed:          int | None = None,
    ) -> np.ndarray:
        """
        Génère un pattern de hi-hat via alpha-blending métrique/anti-métrique.

        Args:
            sync_level:    0 = métrique (on-beat), 1 = mixte, 2 = anti-métrique
            density_level: 0 = sparse, 1 = medium, 2 = dense
            seed:          graine locale (indépendante du RNG global)

        Modèle :
            alpha = sync_level / 2  → 0.0 (métrique pur) à 1.0 (anti-métrique pur)

            struct      = poids_métrique × is_8th_note  (positions paires)
            anti_metric = 1 - poids_métrique            (positions offbeat favorisées)
            prob        = base_density × [(1-alpha)×struct + alpha×anti_metric]

        Le boost ×1.1 sur les 8th notes a été retiré (AMÉLIORATION) car il
        introduisait une asymétrie non justifiée avec le nouveau modèle
        alpha-blend — les 8th notes sont déjà favorisées via struct quand alpha=0.
        """
        rng = np.random.default_rng(seed)
        p   = self.empty()

        # Densité de base
        density_map = {0: 0.30, 1: 0.50, 2: 0.70}
        base_prob   = density_map.get(density_level, 0.50)

        # Pattern 8th notes (positions paires = on-beat)
        eighth_pattern       = np.zeros(self.steps_per_bar)
        eighth_pattern[::2]  = 1.0   # 0, 2, 4, ..., 14

        # Profil métrique normalisé (max → 1.0)
        metric_weight = config.METRIC_PROFILE / config.METRIC_PROFILE.max()

        # Alpha : 0.0 = full métrique, 1.0 = full anti-métrique
        alpha = sync_level / 2.0   # {0: 0.0, 1: 0.5, 2: 1.0}

        # AMÉLIORATION : précalcul des vecteurs pour éviter la boucle Python.
        # Les deux vecteurs sont définis sur une mesure et tuilés.
        struct      = eighth_pattern * metric_weight          # (16,)
        anti_metric = 1.0 - metric_weight                    # (16,)

        # Probabilité par position dans la mesure
        prob_bar = base_prob * ((1.0 - alpha) * struct + alpha * anti_metric)

        # AMÉLIORATION : masque explicite des downbeats forts à alpha élevé.
        # Sans ça, le downbeat (metric_weight=1.0 → anti_metric=0.0) a prob=0
        # à alpha=1 ce qui est correct, mais le temps 3 (0.9 → anti=0.1)
        # reste quasi absent. On laisse le modèle mathématique faire son travail
        # — aucun masque supplémentaire nécessaire avec la normalisation /max().

        prob_bar = np.clip(prob_bar, 0.0, 1.0)

        # Tuilage sur n_steps
        n_bars     = self.n_steps // self.steps_per_bar
        prob_full  = np.tile(prob_bar, n_bars)

        # Tirage stochastique vectorisé
        draws     = rng.random(size=self.n_steps)
        p         = (draws < prob_full).astype(np.float64)

        return p


# =========================================================
# MICRO TIMING
# =========================================================

class MicroTiming:
    """
    Applique des déviations temporelles expressives (jitter) sur un pattern.

    Trois composantes :
        1. Swing     — retard systématique des offbeats (double-croche en grille 16)
        2. Drift     — variation lente sinusoïdale (expressivité globale)
        3. Noise     — bruit gaussien corrélé localement (irrégularité fine)

    Les trois sont pondérées par `amount` (E dans le design) et `voice_weight`
    (hihat > snare > kick, reflétant la hiérarchie perceptive).
    """

    def __init__(self, rng: np.random.Generator, step_duration: float):
        self.rng           = rng
        self.step_duration = step_duration

    def apply(
        self,
        pattern:      np.ndarray,
        amount:       float = 0.0,
        voice_weight: float = 1.0,
    ) -> np.ndarray:
        """
        Args:
            pattern:      pattern binaire (0/1), longueur n_steps
            amount:       intensité globale du jitter (0 = quantisé, 1 = fort)
            voice_weight: poids de la voix (kick=0.3, snare=0.5, hihat=1.0)

        Returns:
            jitters: np.ndarray de décalages en secondes (0 si pas de hit)
        """
        if amount <= 0:
            return np.zeros(len(pattern), dtype=np.float64)

        n    = len(pattern)
        hits = np.where(pattern == 1)[0]

        if len(hits) == 0:
            return np.zeros(n, dtype=np.float64)

        sd = self.step_duration

        # ── 1. Swing ─────────────────────────────────────
        # Retard des positions offbeat (index % 4 == 2 dans une grille 16th)
        # max = 12% du step_duration à amount=1
        swing_strength = 0.12 * amount * sd
        swing          = np.zeros(n)
        swing[2::4]    = swing_strength   # AMÉLIORATION : vectorisé, pas de boucle

        # ── 2. Drift ─────────────────────────────────────
        # Sinusoïde lente (période = 2 × n_steps) avec phase aléatoire
        # max = 10% du step_duration à amount=1
        freq  = 1.0 / (n * 2)
        phase = self.rng.uniform(0, 2 * np.pi)
        drift = np.sin(2 * np.pi * freq * np.arange(n) + phase)
        drift *= 0.10 * amount * sd

        # ── 3. Bruit corrélé ─────────────────────────────
        # Gaussien lissé par un kernel triangulaire (évite les sauts brusques)
        # σ = 10% du step_duration à amount=1
        sigma = 0.10 * amount * sd
        noise = self.rng.normal(0, sigma, size=n)
        noise = np.convolve(noise, np.array([0.25, 0.5, 0.25]), mode="same")

        # ── Combinaison ──────────────────────────────────
        total_shift = swing + drift + noise

        jitters        = np.zeros(n, dtype=np.float64)
        jitters[hits]  = total_shift[hits] * voice_weight

        return jitters


# =========================================================
# STIMULUS
# =========================================================

class Stimulus:
    """Assemble un stimulus complet (patterns + jitters) depuis une config."""

    def __init__(self, voices: Voices, micro: MicroTiming):
        self.voices = voices
        self.micro  = micro

    def build(self, cfg: dict, seed: int) -> dict:
        """
        Args:
            cfg:  dict avec clés S_mv, D_mv, E (+ phase, repeat)
            seed: graine pour reproductibilité du hi-hat

        Returns:
            dict avec kick, snare, hihat, *_jitter, config
        """
        kick  = self.voices.kick()
        snare = self.voices.snare()
        hihat = self.voices.hihat(
            sync_level=cfg["S_mv"],
            density_level=cfg["D_mv"],
            seed=seed,
        )

        # Hiérarchie perceptive : hihat le plus sensible au micro-timing
        kick_j  = self.micro.apply(kick,  amount=cfg["E"] * 0.20, voice_weight=0.3)
        snare_j = self.micro.apply(snare, amount=cfg["E"] * 0.30, voice_weight=0.5)
        hihat_j = self.micro.apply(hihat, amount=cfg["E"] * 1.00, voice_weight=1.0)

        return {
            "kick":         kick,
            "snare":        snare,
            "hihat":        hihat,
            "kick_jitter":  kick_j,
            "snare_jitter": snare_j,
            "hihat_jitter": hihat_j,
            "config":       cfg,
        }


# =========================================================
# METRICS
# =========================================================

class Metrics:
    """
    Métriques acoustiques calculées sur un stimulus assemblé.

    D       — densité globale pondérée (hihat dominant perceptivement)
    I       — variance inter-voix (équilibre kick/snare/hihat)
    V       — variance du micro-timing (dispersion des jitters)
    S_real  — index de syncopation (implémentation proche de Keith 2011)
    E_real  — énergie moyenne du micro-timing (amplitude absolue)
    """

    def global_density(self, stim: dict) -> float:
        """Densité globale pondérée perceptivement (hihat = 60%)."""
        return (
            0.20 * float(np.mean(stim["kick"]))  +
            0.20 * float(np.mean(stim["snare"])) +
            0.60 * float(np.mean(stim["hihat"]))
        )

    def micro_V(self, stim: dict) -> float:
        """Variance des jitters sur toutes les voix (mesure de dispersion)."""
        vals = []
        for v in ["kick", "snare", "hihat"]:
            jitter  = stim[f"{v}_jitter"]
            pattern = stim[v]
            vals.extend(jitter[pattern == 1].tolist())
        return float(np.var(vals)) if vals else 0.0

    def syncopation_index(self, pattern: np.ndarray) -> float:
        """
        Index de syncopation inspiré de Keith (2011).

        Un hit est syncopé si :
          - il tombe sur une position de faible poids métrique
          - ET la prochaine position de poids supérieur n'est pas frappée

        Score = somme des (poids_attendu - poids_réel) pour chaque hit syncopé,
                normalisée par le nombre total de hits.

        Retourne 0.0 si aucun hit.
        """
        n = len(pattern)

        # Profil métrique tuilé sur n_steps
        metric_profile = np.tile(
            config.METRIC_PROFILE,
            int(np.ceil(n / config.STEPS_PER_BAR))
        )[:n]

        hits = np.where(pattern == 1)[0]
        if len(hits) == 0:
            return 0.0

        sync  = 0.0

        for i in hits:
            current_weight = metric_profile[i]

            # Pas syncopé si sur position forte (seuil 0.5)
            if current_weight > 0.5:
                continue

            # Cherche la prochaine position de poids supérieur
            for j in range(i - 1, -1, -1):
                if metric_profile[j] > current_weight :
                    if pattern[j] == 0:
                        sync += (metric_profile[j] - current_weight)

                    break

        return sync / len(hits)

    def inter_voice_variance(self, stim: dict) -> float:
        """Variance des densités entre voix (équilibre rythmique)."""
        densities = [
            float(np.mean(stim["kick"])),
            float(np.mean(stim["snare"])),
            float(np.mean(stim["hihat"])),
        ]
        return float(np.var(densities))

    def micro_E(self, stim: dict) -> float:
        """Énergie moyenne du micro-timing (amplitude absolue des jitters)."""
        vals = []
        for v in ["kick", "snare", "hihat"]:
            jitter  = stim[f"{v}_jitter"]
            pattern = stim[v]
            vals.extend(np.abs(jitter[pattern == 1]).tolist())
        return float(np.mean(vals)) if vals else 0.0


# =========================================================
# DESIGN EXPÉRIMENTAL
# =========================================================

def build_design(n_repeats: int | None = None) -> list[dict]:
    """
    Construit le design factoriel en 3 phases.

    Phase 1 : manipulation de S_mv seul (D_mv=1, E=0) → effet structure
    Phase 2 : manipulation de E seul (S_mv=1, D_mv=1)  → effet micro-timing
    Phase 3 : factoriel complet S × D × E              → interactions

    Chaque condition est répétée `repeats` fois (counterbalancing).
    """
    repeats = config.REPEATS if n_repeats is None else n_repeats

    base = []

    # Phase 1 — syncopation seule
    base += [
        {"phase": 1, "S_mv": s, "D_mv": 1, "E": 0.0}
        for s in config.S_LEVELS
    ]

    # Phase 2 — micro-timing seul
    base += [
        {"phase": 2, "S_mv": 1, "D_mv": 1, "E": e}
        for e in config.E_LEVELS
        if e > 0   # AMÉLIORATION : E=0 est déjà couvert en phase 1 (évite doublon)
    ]

    # Phase 3 — factoriel complet
    base += [
        {"phase": 3, "S_mv": s, "D_mv": d, "E": e}
        for s, d, e in itertools.product(
            config.S_LEVELS,
            config.D_LEVELS,
            config.E_LEVELS,
        )
    ]

    # Expansion avec répétitions
    expanded = [
        {**condition, "repeat": r}
        for condition in base
        for r in range(repeats)
    ]

    return expanded


# =========================================================
# EXPERIMENT
# =========================================================

def run_experiment(
    seed: int | None = None,
    n_repeats: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Lance l'expérience complète : génère tous les stimuli du design.

    Args:
        seed:      graine maître (défaut : config.SEED)
        n_repeats: nombre de répétitions par condition (défaut : config.REPEATS)

    Returns:
        df:    DataFrame avec métriques + patterns pour chaque stimulus
        cache: dict {id: stim_dict} pour export MIDI
    """
    seed = config.SEED if seed is None else seed
    rng  = np.random.default_rng(seed)

    grid    = Grid()
    voices  = Voices(grid, seed=seed)
    micro   = MicroTiming(rng, grid.step_duration)

    stim_builder = Stimulus(voices, micro)
    metrics      = Metrics()

    design = build_design(n_repeats=n_repeats)

    # AMÉLIORATION : shuffle reproductible via le RNG maître
    # (np.random.shuffle ne respecte pas la graine fixée par default_rng)
    indices = rng.permutation(len(design))
    design  = [design[i] for i in indices]

    rows  = {}
    cache = {}

    for i, cfg in enumerate(design):
        # Graine dérivée déterministe : seed + i (reproductible quel que soit i)
        stim = stim_builder.build(cfg, seed=seed + i)
        cache[i] = stim

        rows[i] = {
            # ── Identifiants ──────────────────────────
            "id":     i,
            "phase":  cfg["phase"],
            "repeat": cfg["repeat"],

            # ── Variables manipulées ──────────────────
            "S_mv":  cfg["S_mv"],
            "D_mv":  cfg["D_mv"],
            "E":     cfg["E"],

            # ── Métriques réalisées ───────────────────
            "D":      metrics.global_density(stim),
            "I":      metrics.inter_voice_variance(stim),
            "V":      metrics.micro_V(stim),
            "S_real": metrics.syncopation_index(stim["hihat"]),
            "E_real": metrics.micro_E(stim),

            # ── Contexte ─────────────────────────────
            "BPM":   grid.bpm,

            # ── Patterns bruts (pour export MIDI) ────
            "kick":  stim["kick"].tolist(),
            "snare": stim["snare"].tolist(),
            "hihat": stim["hihat"].tolist(),
        }

    df = pd.DataFrame.from_dict(rows, orient="index")

    return df, cache