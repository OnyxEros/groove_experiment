def describe_cluster(profile):

    desc = []

    # Density
    if profile["density"] < 0.3:
        desc.append("low-density")
    elif profile["density"] < 0.6:
        desc.append("medium-density")
    else:
        desc.append("high-density")

    # Syncopation
    if profile["syncopation"] > 0.6:
        desc.append("strongly syncopated")
    elif profile["syncopation"] > 0.3:
        desc.append("moderately syncopated")
    else:
        desc.append("metrically stable")

    # Micro-timing
    if profile["micro_variance"] > 0.6:
        desc.append("high micro-timing variability")
    elif profile["micro_variance"] > 0.3:
        desc.append("moderate groove looseness")
    else:
        desc.append("tight timing structure")

    # Inter-voice balance
    if profile["inter_voice_var"] > 0.4:
        desc.append("asymmetric voice density")
    else:
        desc.append("balanced drum distribution")

    # Push/pull (optionnel)
    push = profile.get("push_pull", None)
    if push is not None:
        if push > 0.05:
            desc.append("rushing feel (hihat ahead)")
        elif push < -0.05:
            desc.append("laid-back feel (hihat behind)")
        else:
            desc.append("aligned voices")

    return ", ".join(desc)