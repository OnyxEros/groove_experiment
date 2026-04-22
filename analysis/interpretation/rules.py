def describe_cluster(profile):

    desc = []

    # ---------------------------
    # DENSITY
    # ---------------------------
    if profile["density"] < 0.3:
        desc.append("low-density")
    elif profile["density"] < 0.6:
        desc.append("medium-density")
    else:
        desc.append("high-density")

    # ---------------------------
    # SYNCOPATION
    # ---------------------------
    if profile["syncopation"] > 0.6:
        desc.append("strongly syncopated")
    elif profile["syncopation"] > 0.3:
        desc.append("moderately syncopated")
    else:
        desc.append("metrically stable")

    # ---------------------------
    # MICRO TIMING
    # ---------------------------
    if profile["micro_variance"] > 0.6:
        desc.append("high micro-timing variability")
    elif profile["micro_variance"] > 0.3:
        desc.append("moderate groove looseness")
    else:
        desc.append("tight timing structure")

    # ---------------------------
    # INTER VOICE STRUCTURE
    # ---------------------------
    if profile["inter_voice_var"] > 0.4:
        desc.append("asymmetric voice density")
    else:
        desc.append("balanced drum distribution")

    return ", ".join(desc)
