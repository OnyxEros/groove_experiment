"""
groove/generator.py
===================
Générateur de stimuli rythmiques pour l'expérience groove.

Notation :
    Paramètres génératifs (manipulés) : S_mv, D_mv, E_mv, P_mv
    Descripteurs émergents (réalisés)  : D, I, V, S, E, P

Correction v4 :
    G1 — Seeds isolées par voix :
         Avant : hihat(seed=seed+i) où i est l'index global du stimulus.
                 → la seed du hi-hat et celle du MicroTiming partagent le
                   même espace, créant des corrélations implicites entre
                   la structure rythmique et le micro-timing.
                 → deux conditions identiques avec i différents produisent
                   des patterns différents, violant la reproductibilité.
         Après : chaque voix reçoit une seed dérivée de façon déterministe
                 depuis (base_seed, stimulus_index, voice_id) via un hash simple.
                 hihat_seed  = _derive_seed(seed, i, 0)
                 timing_seed = _derive_seed(seed, i, 1)
                 → isolation complète entre voix
                 → reproductibilité garantie pour un (seed, i) donné
                 → inchangé comportement externe (même API)

    G2 — stim_id inclus dans metadata.csv dès la génération :
         Avant : stim_id était ajouté dans run_experiment() mais pas dans
                 build_audio_map() → metadata.csv manquait parfois la colonne.
         Après : stim_id est systématiquement présent dans rows[i] et
                 survivra à df.to_csv().
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

import config


# =========================================================
# SEED ISOLATION
# =========================================================

def _derive_seed(base_seed: int, stim_index: int, voice_id: int) -> int:
    """
    Dérive une seed déterministe et isolée pour une voix d'un stimulus.

    Utilise un hash simple (pas de collisions dans l'espace pratique) :
        seed = (base_seed * 1_000_003 + stim_index * 997 + voice_id) % 2**31

    Propriétés :
        - Déterministe : même (base_seed, stim_index, voice_id) → même seed
        - Isolée : voice_id différent → seeds décorrélées
        - Légère : pas de dépendance à hashlib

    voice_id convention :
        0 → hihat pattern
        1 → micro-timing (MicroTiming.rng)
    """
    return int((base_seed * 1_000_003 + stim_index * 997 + voice_id) % (2 ** 31))


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
        """
        Kick invariant sur tous les stimuli — cadre d'ancrage métrique.
        Beats 1 et 3 uniquement (cf. §2.2.2 du mémoire).
        """
        p, bar = self._empty(), self.steps_per_bar
        for b in range(self.total_steps // bar):
            o = b * bar
            p[o]            = 1.0
            p[o + bar // 2] = 1.0
        return p

    def bass(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Basse invariante sur tous les stimuli (cf. §2.2.2 du mémoire)."""
        bar = self.steps_per_bar
        n   = self.total_steps
        sd  = config.step_duration_seconds()

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
        """Snare invariante — backbeat positions 4 et 12."""
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
        Hi-hat probabiliste calibré perceptivement.

        Args:
            seed : seed ISOLÉE pour ce hi-hat — doit être _derive_seed(base, i, 0)
                   et non seed+i (qui corrèle avec le timing, cf. G1).
        """
        rng       = np.random.default_rng(seed)
        base_prob = config.HIHAT_DENSITY_PROBS.get(density_level, 0.55)
        alpha     = config.alpha_from_sync_level(sync_level)

        metric_weight_bar  = config.METRIC_PROFILE / config.METRIC_PROFILE.max()
        metric_weight_loop = np.tile(metric_weight_bar, config.LOOP_BARS)

        strong    = metric_weight_loop
        weak      = 1.0 - metric_weight_loop
        prob_loop = base_prob * ((1.0 - alpha) * strong + alpha * weak)

        eighth_bias      = np.zeros(self.loop_steps)
        eighth_bias[::2] = 0.10
        prob_loop       += eighth_bias
        prob_loop        = np.clip(prob_loop, config.HIHAT_PROB_MIN, config.HIHAT_PROB_MAX)

        loop_pattern = (rng.random(self.loop_steps) < prob_loop).astype(np.float64)

        max_hits = {0: 5, 1: 10, 2: 15}.get(density_level, 10)
        hits     = np.where(loop_pattern == 1)[0]

        if len(hits) > max_hits:
            to_remove = rng.choice(hits, size=len(hits) - max_hits, replace=False)
            loop_pattern[to_remove] = 0

        pattern = np.tile(loop_pattern, self.n_loops)
        return pattern[:self.total_steps]


# =========================================================
# MICRO TIMING
# =========================================================

class MicroTiming:
    """
    Micro-timing expressif lié à E_mv.

    Le rng interne est initialisé depuis une seed ISOLÉE (_derive_seed(base, i, 1))
    pour éviter toute corrélation avec le hi-hat pattern (cf. G1).
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
        n    = len(pattern)
        hits = np.where(pattern == 1.0)[0]

        if len(hits) == 0:
            return np.zeros(n, dtype=np.float64)

        sd = self.step_duration

        swing_total = (
            config.SWING_BASELINE
            + config.SWING_MAX_RATIO * (amount ** 0.7)
        ) * sd

        swing       = np.zeros(n)
        swing[1::2] = swing_total * 0.6

        if amount > 0.0:
            phase = self.rng.uniform(0.0, 2.0 * np.pi)
            drift = np.sin(2.0 * np.pi * np.arange(n) / (n * 2) + phase)
            drift *= config.DRIFT_MAX_RATIO * amount * sd

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
        n    = len(pattern)
        hits = np.where(pattern == 1.0)[0]

        if len(hits) == 0:
            return np.zeros(n, dtype=np.float64)

        sd = self.step_duration

        anticipation = config.BASS_ANTICIPATION_RATIO * sd
        sigma        = config.BASS_HUMANIZE_NOISE_RATIO * sd
        noise        = self.rng.normal(0.0, sigma, size=n)

        swing_total = (config.SWING_BASELINE + config.SWING_MAX_RATIO * amount) * sd
        swing       = np.zeros(n)
        swing[2::4] = swing_total

        total_shift   = anticipation + noise + swing
        jitters       = np.zeros(n, dtype=np.float64)
        jitters[hits] = total_shift[hits] * config.BASS_VOICE_WEIGHT

        return jitters

    def humanize_velocities(
        self,
        vel_array:  np.ndarray,
        amount:     float,
        sigma_base: float | None = None,
    ) -> np.ndarray:
        if sigma_base is None:
            sigma_base = config.VELOCITY_HUMANIZE_SIGMA

        if amount < 1e-3:
            return vel_array.copy()

        active = vel_array > 0
        result = vel_array.copy().astype(np.float64)

        if active.any():
            noise           = self.rng.normal(
                0.0, sigma_base * amount, size=int(active.sum())
            )
            result[active] += noise
            result[active]  = np.clip(result[active], 1, 127)

        return result


# =========================================================
# STIMULUS
# =========================================================

class Stimulus:
    def __init__(self, voices: Voices, micro: MicroTiming) -> None:
        self.voices = voices
        self.micro  = micro

    def build(self, cfg: dict, seed: int) -> dict:
        """
        Args:
            seed : seed ISOLÉE pour le timing de ce stimulus.
                   Doit être _derive_seed(base_seed, stim_index, 1).
                   Le hi-hat utilise _derive_seed(base_seed, stim_index, 0)
                   — passé séparément via hihat(seed=hihat_seed).
        """
        E_mv  = cfg["E_mv"]
        P_mv  = cfg.get("P_mv", 0)

        hihat_push_s = config.push_from_p_level(P_mv) * self.micro.step_duration

        kick  = self.voices.kick()
        bass, bass_pitch, bass_vel, bass_dur = self.voices.bass()
        snare = self.voices.snare()
        hihat = self.voices.hihat(
            sync_level=cfg["S_mv"],
            density_level=cfg["D_mv"],
            seed=seed,   # ← seed déjà isolée (voice_id=0) passée depuis run_experiment
        )

        kick_j  = self.micro.apply(
            kick,
            amount=E_mv * config.KICK_TIMING_SCALE,
            voice_weight=config.KICK_VOICE_WEIGHT,
        )
        bass_j  = self.micro.apply_bass(bass, amount=E_mv)
        snare_j = self.micro.apply(
            snare,
            amount=E_mv * config.SNARE_TIMING_SCALE,
            voice_weight=config.SNARE_VOICE_WEIGHT,
        )
        hihat_j = self.micro.apply(
            hihat,
            amount=E_mv * config.HIHAT_TIMING_SCALE,
            voice_weight=config.HIHAT_VOICE_WEIGHT,
        )

        hihat_hits          = hihat == 1.0
        hihat_j[hihat_hits] += hihat_push_s

        kick_vel_base  = np.where(kick  == 1.0, 100.0, 0.0)
        snare_vel_base = np.where(snare == 1.0,  95.0, 0.0)
        hihat_vel_base = np.where(hihat == 1.0,  72.0, 0.0)

        kick_vel  = self.micro.humanize_velocities(kick_vel_base,  amount=E_mv)
        snare_vel = self.micro.humanize_velocities(snare_vel_base, amount=E_mv)
        hihat_vel = self.micro.humanize_velocities(hihat_vel_base, amount=E_mv)
        bass_vel  = self.micro.humanize_velocities(bass_vel,       amount=E_mv)

        return {
            "kick":         kick,
            "bass":         bass,
            "bass_pitch":   bass_pitch,
            "bass_vel":     bass_vel,
            "bass_dur":     bass_dur,
            "snare":        snare,
            "hihat":        hihat,
            "kick_vel":     kick_vel,
            "snare_vel":    snare_vel,
            "hihat_vel":    hihat_vel,
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
        """Descripteur émergent S (sans indice)."""
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
        """Descripteur émergent E (sans indice)."""
        vals: list[float] = []
        for v in ("kick", "snare", "hihat"):
            mask = stim[v] == 1.0
            vals.extend(np.abs(stim[f"{v}_jitter"][mask]).tolist())
        return float(np.mean(vals)) if vals else 0.0

    def inter_voice_push(self, stim: dict) -> float:
        """Descripteur émergent P (sans indice) — désalignement inter-voix."""
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
        r1 = r2 = r3 = n_repeats
    else:
        r1 = config.REPEATS_P1
        r2 = config.REPEATS_P2
        r3 = config.REPEATS_P3

    phase1 = [
        {"phase": 1, "S_mv": s, "D_mv": 1, "E_mv": 0.0, "P_mv": 0}
        for s in config.S_mv_LEVELS
    ]
    phase2 = [
        {"phase": 2, "S_mv": 1, "D_mv": 1, "E_mv": e, "P_mv": p}
        for e, p in itertools.product(config.E_mv_LEVELS, config.P_mv_LEVELS)
        if not (e == 0.0 and p == 0)
    ]
    phase3 = [
        {"phase": 3, "S_mv": s, "D_mv": d, "E_mv": e, "P_mv": p}
        for s, d, e, p in itertools.product(
            config.S_mv_LEVELS,
            config.D_mv_LEVELS,
            config.E_mv_LEVELS,
            config.P_mv_LEVELS,
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
    metrics = Metrics(grid.step_duration)

    design = build_design(n_repeats=n_repeats)
    order  = rng.permutation(len(design))
    design = [design[i] for i in order]

    rows:  dict[int, dict] = {}
    cache: dict[int, dict] = {}

    for i, cfg in enumerate(design):
        # ── Seeds isolées par voix (correction G1) ─────────────
        # voice_id=0 : pattern hi-hat (stochastique)
        # voice_id=1 : micro-timing (MicroTiming.rng)
        # Chaque voix a un espace de seed indépendant → pas de corrélation
        # entre la structure rythmique et le micro-timing.
        hihat_seed  = _derive_seed(seed, i, 0)
        timing_seed = _derive_seed(seed, i, 1)

        micro   = MicroTiming(
            rng=np.random.default_rng(timing_seed),
            step_duration=grid.step_duration,
        )
        builder = Stimulus(voices, micro)
        stim    = builder.build(cfg, seed=hihat_seed)

        cache[i] = stim

        rows[i] = {
            "id":      i,
            # G2 — stim_id toujours présent dans metadata.csv
            "stim_id": f"stim_{i:04d}",
            "phase":   cfg["phase"],
            "repeat":  cfg["repeat"],
            # Paramètres génératifs
            "S_mv":    cfg["S_mv"],
            "D_mv":    cfg["D_mv"],
            "E_mv":    cfg["E_mv"],
            "P_mv":    cfg.get("P_mv", 0),
            # Descripteurs émergents
            "D":       metrics.global_density(stim),
            "I":       metrics.inter_voice_variance(stim),
            "V":       metrics.micro_V(stim),
            "S":       metrics.syncopation_index(stim["hihat"]),
            "E":       metrics.micro_E(stim),
            "P":       metrics.inter_voice_push(stim),
            "BPM":     grid.bpm,
            # Patterns (utiles pour PatternEmbedding)
            "kick":    stim["kick"].tolist(),
            "snare":   stim["snare"].tolist(),
            "hihat":   stim["hihat"].tolist(),
        }

    return pd.DataFrame.from_dict(rows, orient="index"), cache