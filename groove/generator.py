"""
groove/generator.py
===================
Générateur de stimuli rythmiques pour l'expérience groove.

Tous les paramètres musicaux et expérimentaux proviennent de config.py.
Ce fichier ne contient aucun magic number.

Changelog v2.1 :
    MicroTiming.apply() :
        Swing baseline (SWING_BASELINE) appliqué indépendamment de E.
        Swing total = SWING_BASELINE + SWING_MAX_RATIO × amount.
        → E=0 sonne "tight humain" plutôt que "séquenceur quantisé".
        → E reste une variable continue et interprétable.
        → Toutes les conditions sonnent musicales.

    Voices.hihat() : génération cyclique (inchangé depuis v2).
    Stimulus.build() : P push/pull (inchangé depuis v2).
    Metrics : P_real (inchangé depuis v2).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

import config


# =========================================================
# GRID
# =========================================================

class Grid:
    def __init__(self) -> None:
        self.steps_per_bar = config.STEPS_PER_BAR
        self.loop_steps    = config.loop_steps()
        self.total_steps   = config.total_steps()
        self.n_loops       = config.N_LOOPS
        self.bpm           = config.BPM
        self.step_duration = config.step_duration_seconds()

    @property
    def n_steps(self) -> int:
        return self.total_steps


# =========================================================
# VOICES
# =========================================================

class Voices:
    def __init__(self, grid: Grid, seed: int | None = None) -> None:
        self.total_steps   = grid.total_steps
        self.steps_per_bar = grid.steps_per_bar
        self.loop_steps    = grid.loop_steps
        self.n_loops       = grid.n_loops
        self._rng          = np.random.default_rng(seed)

    def _empty(self) -> np.ndarray:
        return np.zeros(self.total_steps, dtype=np.float64)

    def kick(self) -> np.ndarray:
        p, bar = self._empty(), self.steps_per_bar
        for b in range(self.total_steps // bar):
            o = b * bar
            p[o]            = 1.0
            p[o + bar // 2] = 1.0
        return p

    def bass(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Ligne de basse déterministe, musicalement crédible.

        Motif fixe sur 1 mesure (16 steps), répété sur toutes les mesures.
        Inclut root, quinte, ghost notes et note d'approche.

        Retourne :
            pattern   : np.ndarray (n_steps,) — hits avec vélocité relative [0,1]
            pitch     : np.ndarray (n_steps,) — pitch MIDI par step
            velocity  : np.ndarray (n_steps,) — vélocité MIDI par step (0–127)
            duration  : np.ndarray (n_steps,) — durée en secondes par step
        """
        bar    = self.steps_per_bar   # 16
        n      = self.total_steps
        sd     = config.step_duration_seconds()

        pat_bar  = np.array(config.BASS_PATTERN_BAR,  dtype=np.float64)
        int_bar  = np.array(config.BASS_INTERVAL_BAR, dtype=np.float64)
        vel_bar  = np.array(config.BASS_VELOCITY_BAR, dtype=np.float64)
        dur_bar  = np.array(config.BASS_DURATION_BAR, dtype=np.float64)

        pattern  = np.tile(pat_bar,  n // bar)[:n]
        pitch    = np.tile(config.BASS_PITCH + int_bar, n // bar)[:n]
        velocity = np.tile(vel_bar,  n // bar)[:n]
        duration = np.tile(dur_bar,  n // bar)[:n] * sd

        return pattern, pitch, velocity, duration

    def snare(self) -> np.ndarray:
        p, bar = self._empty(), self.steps_per_bar
        for b in range(self.total_steps // bar):
            o = b * bar
            p[o + bar // 4]     = 1.0
            p[o + 3 * bar // 4] = 1.0
        return p

    def hihat(
        self,
        sync_level:    int        = 0,
        density_level: int        = 1,
        seed:          int | None = None,
    ) -> np.ndarray:
        """
        Pattern hi-hat stochastique cyclique.
        Généré sur LOOP_BARS mesures, répété N_LOOPS fois.
        """
        rng       = np.random.default_rng(seed)
        base_prob = config.HIHAT_DENSITY_PROBS.get(density_level, 0.50)
        alpha     = config.alpha_from_sync_level(sync_level)

        metric_weight_bar  = config.METRIC_PROFILE / config.METRIC_PROFILE.max()
        metric_weight_loop = np.tile(metric_weight_bar, config.LOOP_BARS)

        eighth_bar      = np.zeros(self.steps_per_bar)
        eighth_bar[::2] = 1.0
        eighth_loop     = np.tile(eighth_bar, config.LOOP_BARS)

        struct    = eighth_loop * metric_weight_loop
        anti      = 1.0 - metric_weight_loop
        prob_loop = base_prob * ((1.0 - alpha) * struct + alpha * anti)
        prob_loop = np.clip(prob_loop, config.HIHAT_PROB_MIN, config.HIHAT_PROB_MAX)
        
        loop_pattern = (rng.random(self.loop_steps) < prob_loop).astype(np.float64)
        pattern      = np.tile(loop_pattern, self.n_loops)
        return pattern[: self.total_steps]


# =========================================================
# MICRO TIMING
# =========================================================

class MicroTiming:
    """
    Applique des déviations temporelles expressives (jitter) sur un pattern.

    Composantes :

        Swing baseline — retard minimal des offbeats, toutes conditions.
                         Amplitude : SWING_BASELINE × step_duration
                         Indépendant de E. Humanise E=0.

        Swing expressif — retard additionnel contrôlé par E.
                          Amplitude : SWING_MAX_RATIO × amount × step_duration

        Swing total = (SWING_BASELINE + SWING_MAX_RATIO × amount) × step_duration

        Drift  — variation lente sinusoïdale (DRIFT_MAX_RATIO × amount × sd)
        Noise  — bruit gaussien corrélé (NOISE_MAX_RATIO × amount × sd)

    Le drift et le noise sont nuls à E=0 (amount=0).
    Seul le swing baseline reste actif à E=0.
    """

    def __init__(self, rng: np.random.Generator, step_duration: float) -> None:
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
            pattern      : pattern binaire (0/1), shape (n_steps,)
            amount       : intensité du jitter = E × timing_scale ∈ [0, 1]
            voice_weight : pondération perceptive de la voix ∈ [0, 1]

        Returns:
            jitters : décalages en secondes, 0 sur les positions sans hit
        """
        n    = len(pattern)
        hits = np.where(pattern == 1.0)[0]

        if len(hits) == 0:
            return np.zeros(n, dtype=np.float64)

        sd = self.step_duration

        # ── Swing (baseline + expressif) ─────────────────────
        # Actif même à amount=0 grâce au baseline.
        # Appliqué sur les offbeats 16th (positions index % 4 == 2).
        swing_total = (config.SWING_BASELINE + config.SWING_MAX_RATIO * amount) * sd
        swing       = np.zeros(n)
        swing[2::4] = swing_total

        # ── Drift et Noise — actifs seulement si amount > 0 ──
        if amount > 0.0:
            # Drift sinusoïdal lent
            phase = self.rng.uniform(0.0, 2.0 * np.pi)
            drift = np.sin(2.0 * np.pi * np.arange(n) / (n * 2) + phase)
            drift *= config.DRIFT_MAX_RATIO * amount * sd

            # Bruit gaussien corrélé
            sigma = config.NOISE_MAX_RATIO * amount * sd
            noise = self.rng.normal(0.0, sigma, size=n)
            noise = np.convolve(noise, np.array([0.25, 0.5, 0.25]), mode="same")
        else:
            drift = np.zeros(n)
            noise = np.zeros(n)

        total_shift   = swing + drift + noise
        jitters       = np.zeros(n, dtype=np.float64)
        jitters[hits] = total_shift[hits] * voice_weight

        return jitters


    def apply_bass(
        self,
        pattern: np.ndarray,
        amount:  float = 0.0,
    ) -> np.ndarray:
        """
        Jitter spécialisé pour la basse.

        Deux composantes indépendantes de E :
            - Anticipation fixe (BASS_ANTICIPATION_RATIO) : simule le
              geste du bassiste qui "plante" légèrement en avance sur
              les temps forts. Constante entre conditions.
            - Bruit gaussien résiduel (BASS_HUMANIZE_NOISE_RATIO) :
              micro-irrégularités d'exécution, amplitude fixe et faible.

        Une composante dépendante de E :
            - Swing baseline hérité du kick (amount × BASS_TIMING_SCALE),
              cohérent avec l'humanisation globale du pattern.

        La basse suit E pour rester cohérente avec le feel rythmique global,
        mais garde une personnalité propre via l'anticipation fixe.
        """
        n    = len(pattern)
        hits = np.where(pattern == 1.0)[0]

        if len(hits) == 0:
            return np.zeros(n, dtype=np.float64)

        sd = self.step_duration

        # ── Anticipation fixe (indépendante de E) ─────────────
        anticipation = config.BASS_ANTICIPATION_RATIO * sd   # légèrement négatif = avance

        # ── Bruit résiduel fixe ───────────────────────────────
        sigma = config.BASS_HUMANIZE_NOISE_RATIO * sd
        noise = self.rng.normal(0.0, sigma, size=n)

        # ── Swing cohérent avec le groove global ──────────────
        swing_total = (config.SWING_BASELINE + config.SWING_MAX_RATIO * amount) * sd
        swing       = np.zeros(n)
        swing[2::4] = swing_total

        total_shift   = anticipation + noise + swing
        jitters       = np.zeros(n, dtype=np.float64)
        jitters[hits] = total_shift[hits] * config.BASS_VOICE_WEIGHT

        return jitters


# =========================================================
# STIMULUS
# =========================================================

class Stimulus:
    """
    Assemble un stimulus complet.
    P (push/pull inter-voix) appliqué comme décalage uniforme sur le hihat.
    """

    def __init__(self, voices: Voices, micro: MicroTiming) -> None:
        self.voices = voices
        self.micro  = micro

    def build(self, cfg: dict, seed: int) -> dict:
        E       = cfg["E"]
        P_level = cfg.get("P", 0)

        hihat_push_s = config.push_from_p_level(P_level) * self.micro.step_duration

        kick  = self.voices.kick()
        bass, bass_pitch, bass_vel, bass_dur = self.voices.bass()
        snare = self.voices.snare()
        hihat = self.voices.hihat(
            sync_level=cfg["S_mv"],
            density_level=cfg["D_mv"],
            seed=seed,
        )

        kick_j  = self.micro.apply(
            kick,
            amount=E * config.KICK_TIMING_SCALE,
            voice_weight=config.KICK_VOICE_WEIGHT,
        )
        bass_j = self.micro.apply(
            bass,
            amount=E * config.BASS_TIMING_SCALE,
        )
        snare_j = self.micro.apply(
            snare,
            amount=E * config.SNARE_TIMING_SCALE,
            voice_weight=config.SNARE_VOICE_WEIGHT,
        )
        hihat_j = self.micro.apply(
            hihat,
            amount=E * config.HIHAT_TIMING_SCALE,
            voice_weight=config.HIHAT_VOICE_WEIGHT,
        )

        # Push/pull : décalage uniforme sur tous les hits du hihat
        hihat_hits      = hihat == 1.0
        hihat_j[hihat_hits] += hihat_push_s

        return {
            "kick":         kick,
            "bass":         bass,
            "bass_pitch":   bass_pitch,
            "bass_vel":     bass_vel,
            "bass_dur":     bass_dur,
            "snare":        snare,
            "hihat":        hihat,
            "kick_jitter":  kick_j,
            "bass_jitter":  bass_j, 
            "snare_jitter": snare_j,
            "hihat_jitter": hihat_j,
            "hihat_push":   hihat_push_s,
            "config":       cfg,
        }


# =========================================================
# METRICS
# =========================================================

class Metrics:
    def __init__(self, step_duration: float) -> None:
        self.step_duration = step_duration

    def global_density(self, stim: dict) -> float:
        return (
            config.KICK_DENSITY_WEIGHT  * float(np.mean(stim["kick"]))  +
            config.SNARE_DENSITY_WEIGHT * float(np.mean(stim["snare"])) +
            config.HIHAT_DENSITY_WEIGHT * float(np.mean(stim["hihat"]))
        )

    def micro_V(self, stim: dict) -> float:
        vals: list[float] = []
        for v in ("kick", "snare", "hihat"):
            mask = stim[v] == 1.0
            vals.extend(stim[f"{v}_jitter"][mask].tolist())
        return float(np.var(vals)) if vals else 0.0

    def syncopation_index(self, pattern: np.ndarray) -> float:
        n      = len(pattern)
        metric = np.tile(
            config.METRIC_PROFILE,
            int(np.ceil(n / config.STEPS_PER_BAR))
        )[:n]
        hits = np.where(pattern == 1.0)[0]
        if len(hits) == 0:
            return 0.0
        threshold  = config.SYNCOPATION_STRONG_THRESHOLD
        sync_score = 0.0
        for i in hits:
            w_i = metric[i]
            if w_i > threshold:
                continue
            for j in range(i - 1, -1, -1):
                if metric[j] > w_i:
                    if pattern[j] == 0.0:
                        sync_score += metric[j] - w_i
                    break
        return sync_score / len(hits)

    def inter_voice_variance(self, stim: dict) -> float:
        densities = [
            float(np.mean(stim["kick"])),
            float(np.mean(stim["snare"])),
            float(np.mean(stim["hihat"])),
        ]
        return float(np.var(densities))

    def micro_E(self, stim: dict) -> float:
        vals: list[float] = []
        for v in ("kick", "snare", "hihat"):
            mask = stim[v] == 1.0
            vals.extend(np.abs(stim[f"{v}_jitter"][mask]).tolist())
        return float(np.mean(vals)) if vals else 0.0

    def inter_voice_push(self, stim: dict) -> float:
        """P_real — désalignement inter-voix mesuré, signé, normalisé."""
        hihat_hits  = np.where(stim["hihat"] == 1.0)[0]
        anchor_hits = np.where(
            (stim["kick"] == 1.0) | (stim["snare"] == 1.0)
        )[0]
        if len(hihat_hits) == 0 or len(anchor_hits) == 0:
            return 0.0

        anchor_j = np.zeros(len(stim["kick"]), dtype=np.float64)
        for v, key in [("kick", "kick_jitter"), ("snare", "snare_jitter")]:
            mask = stim[v] == 1.0
            anchor_j[mask] = stim[key][mask]

        window = 2
        diffs: list[float] = []
        for hi in hihat_hits:
            distances = np.abs(anchor_hits - hi)
            if distances.min() <= window:
                closest = anchor_hits[distances.argmin()]
                diffs.append(
                    float(stim["hihat_jitter"][hi])
                    - float(anchor_j[closest])
                )
        if not diffs:
            return 0.0
        sd = self.step_duration
        return float(np.mean(diffs) / sd) if sd > 0 else 0.0


# =========================================================
# DESIGN EXPÉRIMENTAL
# =========================================================

def build_design(n_repeats: int | None = None) -> list[dict]:

    if n_repeats is not None:
        # override CLI (--repeats N) → toutes les phases au même niveau
        r1 = r2 = r3 = n_repeats
    else:
        r1 = config.REPEATS_P1
        r2 = config.REPEATS_P2
        r3 = config.REPEATS_P3

    phase1 = [
        {"phase": 1, "S_mv": s, "D_mv": 1, "E": 0.0, "P": 0}
        for s in config.S_LEVELS
    ]
    phase2 = [
        {"phase": 2, "S_mv": 1, "D_mv": 1, "E": e, "P": p}
        for e, p in itertools.product(config.E_LEVELS, config.P_LEVELS)
        if not (e == 0.0 and p == 0)
    ]
    phase3 = [
        {"phase": 3, "S_mv": s, "D_mv": d, "E": e, "P": p}
        for s, d, e, p in itertools.product(
            config.S_LEVELS,
            config.D_LEVELS,
            config.E_LEVELS,
            config.P_LEVELS,
        )
    ]

    return (
        [{**c, "repeat": r} for c in phase1 for r in range(r1)] +
        [{**c, "repeat": r} for c in phase2 for r in range(r2)] +
        [{**c, "repeat": r} for c in phase3 for r in range(r3)]
    )


# =========================================================
# EXPERIMENT
# =========================================================

def run_experiment(
    seed:      int | None = None,
    n_repeats: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    seed = config.SEED if seed is None else seed
    rng  = np.random.default_rng(seed)

    grid    = Grid()
    voices  = Voices(grid, seed=seed)
    micro   = MicroTiming(rng, grid.step_duration)
    builder = Stimulus(voices, micro)
    metrics = Metrics(grid.step_duration)

    design = build_design(n_repeats=n_repeats)
    order  = rng.permutation(len(design))
    design = [design[i] for i in order]

    rows:  dict[int, dict] = {}
    cache: dict[int, dict] = {}

    for i, cfg in enumerate(design):
        stim     = builder.build(cfg, seed=seed + i)
        cache[i] = stim

        rows[i] = {
            "id":      i,
            "stim_id": f"stim_{i:04d}",
            "phase":   cfg["phase"],
            "repeat":  cfg["repeat"],
            "S_mv":    cfg["S_mv"],
            "D_mv":    cfg["D_mv"],
            "E":       cfg["E"],
            "P":       cfg.get("P", 0),
            "D":       metrics.global_density(stim),
            "I":       metrics.inter_voice_variance(stim),
            "V":       metrics.micro_V(stim),
            "S_real":  metrics.syncopation_index(stim["hihat"]),
            "E_real":  metrics.micro_E(stim),
            "P_real":  metrics.inter_voice_push(stim),
            "BPM":     grid.bpm,
            "kick":    stim["kick"].tolist(),
            "snare":   stim["snare"].tolist(),
            "hihat":   stim["hihat"].tolist(),
        }

    return pd.DataFrame.from_dict(rows, orient="index"), cache