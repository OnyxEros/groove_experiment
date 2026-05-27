import numpy as np


class DiagnosticsCollector:

    def _safe_mean(self, x):
        return float(np.mean(x)) if len(x) else None

    def _safe_max(self, x):
        return float(np.max(x)) if len(x) else None

    # =========================
    # CLUSTERING
    # =========================
    def attach_clustering(self, ctx, metrics: dict):

        ctx.diagnostics["clustering"] = {
            "type": "clustering",
            "silhouette": metrics.get("silhouette"),
            "calinski_harabasz": metrics.get("calinski_harabasz"),
            "davies_bouldin": metrics.get("davies_bouldin"),
            "inertia": metrics.get("inertia"),
            "n_clusters": metrics.get("n_clusters"),
            "cluster_sizes": metrics.get("cluster_sizes"),
        }

    # =========================
    # CORRELATION STRUCTURE
    # =========================
    def attach_correlation(self, ctx, corr):

        abs_corr = np.abs(corr)

        ctx.diagnostics["correlation"] = {
            "type": "correlation",
            "mean_abs": float(np.mean(abs_corr)),
            "max": float(np.max(abs_corr)),
            "min": float(np.min(abs_corr)),
            "strong_ratio": float(np.mean(abs_corr > 0.5)),
            "structure": (
                "highly_correlated"
                if np.mean(abs_corr > 0.5) > 0.3
                else "moderate"
            ),
        }

    # =========================
    # MULTICOLLINEARITY (VIF)
    # =========================
    def attach_vif(self, ctx, names, values):

        vals = np.array(values)

        ctx.diagnostics["vif"] = {
            "type": "vif",
            "mean": float(np.mean(vals)),
            "max": float(np.max(vals)),
            "over_10_ratio": float(np.mean(vals > 10)),
            "over_5_ratio": float(np.mean(vals > 5)),
            "risk_level": (
                "high" if np.mean(vals > 10) > 0.2
                else "moderate" if np.mean(vals > 5) > 0.3
                else "low"
            ),
        }

    # =========================
    # STABILITY / CV
    # =========================
    def attach_cv(self, ctx, cv_data):

        # FIX #6 : np.concatenate crash si cv_data est vide ou ne contient que
        # des tableaux vides. Guard explicite + fallback sur valeurs neutres.
        arrays = [np.array(x) for x in cv_data if len(x) > 0]
        if not arrays:
            ctx.diagnostics["reproducibility"] = {
                "type": "cv",
                "cv_mean": None,
                "cv_median": None,
                "instability_ratio": None,
                "stability": "unknown",
            }
            return

        flat = np.concatenate(arrays)

        ctx.diagnostics["reproducibility"] = {
            "type": "cv",
            "cv_mean": float(np.mean(flat)),
            "cv_median": float(np.median(flat)),
            "instability_ratio": float(np.mean(flat > 0.25)),
            "stability": (
                "unstable" if np.mean(flat > 0.25) > 0.3
                else "moderate" if np.mean(flat > 0.15) > 0.2
                else "stable"
            ),
        }

    # =========================
    # GLOBAL SUMMARY
    # =========================
    def finalize(self, ctx):

        summary = {
            "has_clustering": "clustering" in ctx.diagnostics,
            "has_correlation": "correlation" in ctx.diagnostics,
            "has_vif": "vif" in ctx.diagnostics,
            "has_cv": "reproducibility" in ctx.diagnostics,
        }

        ctx.diagnostics["summary"] = summary

        return ctx.diagnostics