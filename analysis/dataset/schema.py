# analysis/dataset/schema.py

REQUIRED_COLUMNS = [
    "id",
    "phase",
    "repeat",
    "S_mv",
    "D_mv",
    "E",
    "P",        # push/pull inter-voix (paramètre génératif)
    "D",
    "I",
    "V",
    "S_real",
    "E_real",
    "P_real",   # désalignement inter-voix mesuré
    "BPM",
]