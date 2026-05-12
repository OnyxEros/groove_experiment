import numpy as np


class ClusterProfileBuilder:

    def build(self, df, labels):
        profiles = {}

        for c in np.unique(labels):
            mask   = labels == c
            subset = df[mask]

            profile = {
                "size":           int(mask.sum()),
                "density":        float(subset["D"].mean()),
                "syncopation":    float(subset["S"].mean()),        # émergent S
                "micro_variance": float(subset["V"].mean()),
                "inter_voice_var": float(subset["I"].mean()),
                # Paramètres génératifs — accès sécurisé avec get() ou guard
                "S_mv": float(subset["S_mv"].mean()) if "S_mv" in subset.columns else None,
                "D_mv": float(subset["D_mv"].mean()) if "D_mv" in subset.columns else None,
                "E_mv": float(subset["E_mv"].mean()) if "E_mv" in subset.columns else None,
            }

            # P_mv (génératif) — optionnel
            if "P_mv" in subset.columns:
                profile["P_mv"] = float(subset["P_mv"].mean())

            # P (émergent push/pull réalisé) — optionnel
            if "P" in subset.columns:
                profile["push_pull"] = float(subset["P"].mean())

            # E (émergent micro-timing réalisé) — distinct de E_mv
            if "E" in subset.columns:
                profile["micro_timing_realized"] = float(subset["E"].mean())

            profiles[c] = profile

        return profiles