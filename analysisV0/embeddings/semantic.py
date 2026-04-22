def semantic_label_groove(row):
    """
    Convert numeric cluster stats → musical meaning.
    """

    D = row["D_mean"]
    V = row["V_mean"]
    S = row["S_real_mean"]

    # heuristics (simple but effective)

    if D > 0.7 and S > 0.6:
        return "dense / stable groove"

    if V > 0.7:
        return "unstable / high variability"

    if D < 0.3:
        return "sparse / minimal rhythm"

    return "neutral groove texture"