import numpy as np


class ClusterProfileBuilder:

    def build(self, df, labels):
        profiles = {}

        for c in np.unique(labels):
            mask   = labels == c
            subset = df[mask]

            profile = {
                "size":           int(mask.sum()),
                "density":        subset["D"].mean(),
                "syncopation":    subset["S_real"].mean(),
                "micro_variance": subset["V"].mean(),
                "inter_voice_var": subset["I"].mean(),
                "S_mv":           subset["S_mv"].mean(),
                "D_mv":           subset["D_mv"].mean(),
                "E":              subset["E"].mean(),
            }

            # P et P_real — optionnels (rétro-compatibilité)
            if "P" in subset.columns:
                profile["P"] = subset["P"].mean()
            if "P_real" in subset.columns:
                profile["push_pull"] = subset["P_real"].mean()

            profiles[c] = profile

        return profiles