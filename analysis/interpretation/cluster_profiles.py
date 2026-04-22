import numpy as np


class ClusterProfileBuilder:

    def build(self, df, labels):

        profiles = {}

        for c in np.unique(labels):

            mask = labels == c
            subset = df[mask]

            profiles[c] = {
                "size": int(mask.sum()),

                # rhythm structure
                "density": subset["D"].mean(),
                "syncopation": subset["S_real"].mean(),

                # micro structure
                "micro_variance": subset["V"].mean(),
                "inter_voice_var": subset["I"].mean(),

                # generative params
                "S_mv": subset["S_mv"].mean(),
                "D_mv": subset["D_mv"].mean(),
                "E": subset["E"].mean(),
            }

        return profiles
