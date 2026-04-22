import numpy as np
from analysis.interpretation.rules import describe_cluster


def _to_py(x):
    if isinstance(x, (np.generic,)):
        return x.item()
    return x


class ClusterInterpreter:

    def interpret(self, profiles):

        semantic_map = {}

        for c, profile in profiles.items():

            cid = int(c) if isinstance(c, (np.integer, np.floating)) else c

            safe_profile = {k: _to_py(v) for k, v in profile.items()}

            semantic_map[cid] = {
                "profile": safe_profile,
                "label": describe_cluster(safe_profile)
            }

        return semantic_map