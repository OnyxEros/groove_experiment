import numpy as np
import itertools
import pandas as pd

import config


# =========================================================
# GRID
# =========================================================

class Grid:
    def __init__(self):
        self.steps_per_bar = config.STEPS_PER_BAR
        self.n_steps = self.steps_per_bar * config.BARS

        self.bpm = config.BPM

        # durée d'un step
        self.step_duration = 60 / (self.bpm * (self.steps_per_bar / 4))


# =========================================================
# VOICES
# =========================================================

class Voices:
    def __init__(self, grid: Grid, seed=None):
        self.n_steps = grid.n_steps
        self.steps_per_bar = grid.steps_per_bar
        self.rng = np.random.default_rng(seed)

    def empty(self):
        return np.zeros(self.n_steps)

    def kick(self):
        p = self.empty()
        bar = self.steps_per_bar

        for b in range(self.n_steps // bar):
            o = b * bar
            p[o] = 1
            p[o + bar // 2] = 1

        return p

    def snare(self):
        p = self.empty()
        bar = self.steps_per_bar

        for b in range(self.n_steps // bar):
            o = b * bar
            p[o + bar // 4] = 1
            p[o + 3 * bar // 4] = 1

        return p

    def hihat(self, sync_level=0, density_level=1, seed=None):

        rng = np.random.default_rng(seed)
        p = self.empty()

        # D_mv contrôle directement la probabilité de tirage
        density_map = {0: 0.3, 1: 0.5, 2: 0.7}
        base_prob = density_map.get(density_level, 0.5)

        pattern = np.zeros(self.steps_per_bar)
        pattern[::2] = 1  # 8th notes

        for i in range(self.n_steps):

            base = pattern[i % self.steps_per_bar]
            prob = base_prob

            if sync_level == 1:
                prob *= (0.7 if i % 4 == 0 else 1.0)
            elif sync_level == 2:
                prob *= (1.2 if i % 3 == 0 else 0.8)

            if base == 1:
                prob *= 1.4

            if rng.random() < prob:
                p[i] = 1

        return p


# =========================================================
# MICRO TIMING (CORRIGÉ)
# =========================================================

class MicroTiming:
    def __init__(self, rng, step_duration):
        self.rng = rng
        self.step_duration = step_duration

    def apply(self, pattern, amount=0.0, voice_weight=1.0):

        if amount <= 0:
            return np.zeros(len(pattern))

        n = len(pattern)
        jitters = np.zeros(n)
        hits = np.where(pattern == 1)[0]

        # =====================================================
        # 1. SWING MUSICAL (offbeat réel)
        # =====================================================

        swing_strength = 0.12 * amount  # max 12% du step
        swing = np.zeros(n)

        for i in range(n):
            if (i % 4) == 2:  # offbeat (16th grid)
                swing[i] = swing_strength * self.step_duration

        # =====================================================
        # 2. DRIFT LENT (contrôlé)
        # =====================================================

        freq = 1.0 / (n * 2)  # variation lente sur 2 cycles
        phase = self.rng.uniform(0, 2 * np.pi)

        drift = np.sin(2 * np.pi * freq * np.arange(n) + phase)
        drift *= 0.1 * amount * self.step_duration

        # =====================================================
        # 3. BRUIT LOCAL CORRÉLÉ (léger)
        # =====================================================

        sigma = 0.1 * amount * self.step_duration
        noise = self.rng.normal(0, sigma, size=n)

        kernel = np.array([0.25, 0.5, 0.25])
        noise = np.convolve(noise, kernel, mode="same")

        # =====================================================
        # COMBINAISON
        # =====================================================

        total_shift = swing + drift + noise

        for i in hits:
            jitters[i] = total_shift[i] * voice_weight

        return jitters


# =========================================================
# STIMULUS
# =========================================================

class Stimulus:
    def __init__(self, voices, micro):
        self.voices = voices
        self.micro = micro

    def build(self, cfg, seed):

        kick = self.voices.kick()
        snare = self.voices.snare()

        hihat = self.voices.hihat(
            sync_level=cfg["S_mv"],
            density_level=cfg["D_mv"],
            seed=seed
        )

        # 🔥 hiérarchie musicale respectée
        kick_j = self.micro.apply(kick, cfg["E"] * 0.2, 0.3)
        snare_j = self.micro.apply(snare, cfg["E"] * 0.3, 0.5)
        hihat_j = self.micro.apply(hihat, cfg["E"], 1.0)

        return {
            "kick": kick,
            "snare": snare,
            "hihat": hihat,
            "kick_jitter": kick_j,
            "snare_jitter": snare_j,
            "hihat_jitter": hihat_j,
            "config": cfg
        }


# =========================================================
# METRICS
# =========================================================

class Metrics:

    def global_density(self, stim):
        return (
            0.2 * np.mean(stim["kick"]) +
            0.2 * np.mean(stim["snare"]) +
            0.6 * np.mean(stim["hihat"])
        )

    def micro_V(self, stim):
        vals = []

        for v in ["kick", "snare", "hihat"]:
            vals.extend(stim[f"{v}_jitter"][stim[v] == 1])

        return np.var(vals) if len(vals) > 0 else 0.0

    def syncopation_index(self, pattern):
        n = len(pattern)

        # hiérarchie métrique simple (4/4 en 16 steps)
        metric_profile = np.array([
            1.0, 0.2, 0.6, 0.2,
            0.8, 0.2, 0.5, 0.2,
            0.9, 0.2, 0.6, 0.2,
            0.7, 0.2, 0.5, 0.2
        ])

        metric_profile = np.tile(metric_profile, n // config.STEPS_PER_BAR)

        # attente = proba métrique normalisée
        p = metric_profile / np.max(metric_profile)

        # coût syncopation = écart à l'attendu
        return np.mean(np.abs(pattern - p))


    def inter_voice_variance(self, stim):
        densities = [
            np.mean(stim["kick"]),
            np.mean(stim["snare"]),
            np.mean(stim["hihat"])
        ]
        return np.var(densities)


    def micro_E(self, stim):
        vals = []

        for v in ["kick", "snare", "hihat"]:
            vals.extend(np.abs(stim[f"{v}_jitter"][stim[v] == 1]))

        return np.mean(vals) if len(vals) > 0 else 0.0


# =========================================================
# DESIGN
# =========================================================

def build_design(n_repeats=None):

    repeats = config.REPEATS if n_repeats is None else n_repeats

    base = []

    base += [
        {"phase": 1, "S_mv": s, "D_mv": 1, "E": 0.0}
        for s in config.S_LEVELS
    ]

    base += [
        {"phase": 2, "S_mv": 1, "D_mv": 1, "E": e}
        for e in config.E_LEVELS
    ]

    base += [
        {"phase": 3, "S_mv": s, "D_mv": d, "E": e}
        for s, d, e in itertools.product(
            config.S_LEVELS,
            config.D_LEVELS,
            config.E_LEVELS
        )
    ]

    expanded = []
    for d in base:
        for r in range(repeats):
            expanded.append({**d, "repeat": r})

    return expanded


# =========================================================
# EXPERIMENT
# =========================================================

def run_experiment(seed=None, n_repeats=None):

    seed = config.SEED if seed is None else seed
    rng = np.random.default_rng(seed)

    grid = Grid()
    voices = Voices(grid, seed=seed)
    micro = MicroTiming(rng, grid.step_duration)

    stim_builder = Stimulus(voices, micro)
    metrics = Metrics()

    design = build_design(n_repeats=n_repeats)
    rng.shuffle(design)

    rows = []
    cache = {}

    for i, cfg in enumerate(design):

        stim = stim_builder.build(cfg, seed + i)
        cache[i] = stim

        rows.append({
            "id": i,
            "phase": cfg["phase"],
            "repeat": cfg["repeat"],
            "S_mv": cfg["S_mv"],
            "D_mv": cfg["D_mv"],
            "E": cfg["E"],

            "D": metrics.global_density(stim),
            "I": metrics.inter_voice_variance(stim),
            "V": metrics.micro_V(stim),
            "S_real": metrics.syncopation_index(stim["hihat"]),
            "E_real": metrics.micro_E(stim),

            "BPM": grid.bpm,

            "kick": stim["kick"].tolist(),
            "snare": stim["snare"].tolist(),
            "hihat": stim["hihat"].tolist(),
        })

    return pd.DataFrame(rows), cache