from backend.design.conditions import generate_conditions, DEFAULT_SPACE
from backend.design.sampler import sample_variants
from groove.generator import generate_groove


class StimulusRegistry:

    def __init__(self, space=DEFAULT_SPACE):
        self.space = space
        self.conditions = None

    def build(self, seed=42):
        self.conditions = generate_conditions(self.space, seed=seed)
        return self.conditions

    def build_stimuli(self, n_variants=3, seed=42):
        if self.conditions is None:
            self.build(seed)

        stimuli = []

        for i, cond in enumerate(self.conditions):
            variants = sample_variants(
                cond,
                generator_fn=generate_groove,
                n_variants=n_variants,
                base_seed=i * 1000
            )

            for v in variants:
                stimuli.append(v)

        return stimuli
