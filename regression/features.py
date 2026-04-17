def add_features(df):
    """
    Add model variables (theoretical + emergent)
    """

    # ensure numeric stability
    df["groove"] = df["groove"].astype(float)

    # interactions / non-linear terms (paper-ready)
    df["S_mv_sq"] = df.get("S_mv", 0) ** 2

    # placeholders if not yet computed
    if "D" not in df:
        df["D"] = 0
    if "I" not in df:
        df["I"] = 0
    if "V" not in df:
        df["V"] = 0

    return df