# =========================
# CORE
# =========================

import numpy as np
import itertools
import pandas as pd


# =========================
# GRID
# =========================

class Grid:
    def __init__(self, n_steps=16, bpm=120, subdivision=4, bars=8):
        self.n_steps = n_steps * bars
        self.bpm = bpm
        self.subdivision = subdivision
        self.step_duration = 60 / (bpm * subdivision)


# =========================
# VOICES
# =========================

class Voices:
    def __init__(self, n_steps=16, seed=None):
        self.n_steps = n_steps
        self.rng = np.random.default_rng(seed)

    def empty(self):
        return np.zeros(self.n_steps)

    def kick(self):
        p = self.empty()

        bar = 16
        n_bars = self.n_steps // bar

        for b in range(n_bars):
            offset = b * bar
            p[offset + 0] = 1
            p[offset + 8] = 1

        return p

    def snare(self):
        p = self.empty()

        bar = 16
        n_bars = self.n_steps // bar

        for b in range(n_bars):
            offset = b * bar
            p[offset + 4] = 1
            p[offset + 12] = 1

        return p

    def hihat(self, sync_level=0, density_level=1, seed=None):

        rng = np.random.default_rng(seed)
        p = self.empty()

        density_map = {0: 0.3, 1: 0.55, 2: 0.75}
        target_density = density_map.get(density_level, 0.55)

        cycle = 16
        n_cycles = self.n_steps // cycle

        base_pattern = np.tile([1, 0, 1, 0], 4)

        for bar in range(n_cycles):

            offset = bar * cycle

            for i in range(cycle):

                idx = offset + i
                if idx >= self.n_steps:
                    continue

                if sync_level == 0:
                    prob = target_density
                elif sync_level == 1:
                    prob = target_density * (0.7 if i % 4 == 0 else 1.0)
                else:
                    prob = target_density * (1.2 if i % 3 == 0 else 0.8)

                if base_pattern[i % len(base_pattern)] == 1:
                    prob += 0.2

                if rng.random() < prob:
                    p[idx] = 1

        return p


# =========================
# MICRO TIMING
# =========================

class MicroTiming:
    def __init__(self, rng, step_duration):
        self.rng = rng
        self.step_duration = step_duration

    def apply(self, pattern, amount=0.0, mode="gaussian", voice_weight=1.0):

        if amount == 0 or mode == "none":
            return np.zeros(len(pattern))

        jitters = np.zeros(len(pattern))
        hits = np.where(pattern == 1)[0]

        for i in hits:

            bar_drift = self.rng.normal(0, amount) * self.step_duration
            local = self.rng.normal(0, amount * 0.2) * self.step_duration

            jitter = (bar_drift + local) * voice_weight

            jitters[i] = np.clip(
                jitter,
                -0.4 * self.step_duration,
                0.4 * self.step_duration
            )

        return jitters


# =========================
# STIMULUS
# =========================

class Stimulus:
    def __init__(self, voices, micro):
        self.voices = voices
        self.micro = micro

    def build(self, config, seed=None):

        rng_seed = seed if seed is not None else 0

        kick = self.voices.kick()
        snare = self.voices.snare()

        hihat = self.voices.hihat(
            sync_level=config["S_mv"],
            density_level=config["D_mv"],
            seed=rng_seed
        )

        kick_j = self.micro.apply(kick, config["E"], config["micro_mode"], 0.5)
        snare_j = self.micro.apply(snare, config["E"], config["micro_mode"], 0.7)
        hihat_j = self.micro.apply(hihat, config["E"], config["micro_mode"], 1.0)

        return {
            "kick": kick,
            "snare": snare,
            "hihat": hihat,
            "kick_jitter": kick_j,
            "snare_jitter": snare_j,
            "hihat_jitter": hihat_j,
            "config": config
        }


# =========================
# METRICS
# =========================

class Metrics:

    def global_density(self, stim):
        return (
            0.2 * np.mean(stim["kick"]) +
            0.2 * np.mean(stim["snare"]) +
            0.6 * np.mean(stim["hihat"])
        )

    def micro_V(self, stim):
        jitters = []

        for v in ["kick", "snare", "hihat"]:
            mask = stim[v] == 1
            jitters.extend(stim[f"{v}_jitter"][mask])

        return np.var(jitters) if len(jitters) > 0 else 0.0

    def syncopation_index(self, pattern):
        weights = np.tile([3, 1, 2, 1], len(pattern) // 4)

        weights = np.where(weights == 0, 1, weights)
        return np.mean(pattern / weights)

    def inter_voice_variance(self, stim):
        densities = [
            np.mean(stim["kick"]),
            np.mean(stim["snare"]),
            np.mean(stim["hihat"])
        ]
        return np.var(densities)


# =========================
# DESIGN
# =========================

def build_design(seed=42, n_repeats=8):

    S_levels = [0, 1, 2]
    D_levels = [0, 1, 2]
    E_levels = [0.0, 0.5, 1.0]

    base = []

    base += [
        {"phase": 1, "S_mv": S, "D_mv": 1, "E": 0.0, "micro_mode": "none"}
        for S in S_levels
    ]

    base += [
        {"phase": 2, "S_mv": 1, "D_mv": 1, "E": E, "micro_mode": "gaussian"}
        for E in E_levels
    ]

    base += [
        {"phase": 3, "S_mv": S, "D_mv": D, "E": E, "micro_mode": "gaussian"}
        for S, D, E in itertools.product(S_levels, D_levels, E_levels)
    ]

    expanded = []
    for d in base:
        for r in range(n_repeats):
            d2 = d.copy()
            d2["repeat"] = r
            expanded.append(d2)

    return expanded


# =========================
# PIPELINE
# =========================

def run_experiment(seed=42, n_repeats=8):

    rng = np.random.default_rng(seed)

    grid = Grid()
    voices = Voices(seed=seed)
    micro = MicroTiming(rng, grid.step_duration)
    stim_builder = Stimulus(voices, micro)
    metrics = Metrics()

    design = build_design(seed, n_repeats)
    rng.shuffle(design)

    rows = []
    stim_cache = {}

    for i, config in enumerate(design):

        stim = stim_builder.build(config, seed=seed + i)
        stim_cache[i] = stim

        rows.append({
            "id": i,
            "phase": config["phase"],
            "repeat": config["repeat"],
            "S_mv": config["S_mv"],
            "D_mv": config["D_mv"],
            "E": config["E"],
            "D": metrics.global_density(stim),
            "I": metrics.inter_voice_variance(stim),
            "V": metrics.micro_V(stim),
            "S_real": metrics.syncopation_index(stim["hihat"]),
            "BPM": grid.bpm
        })

    return pd.DataFrame(rows), stim_cache