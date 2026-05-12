# analysis/dataset/schema.py
#
# Notation :
#   Paramètres génératifs (manipulés) : S_mv, D_mv, E_mv, P_mv
#   Descripteurs émergents (réalisés)  : D, I, V, S, E, P

REQUIRED_COLUMNS = [
    "id",
    "phase",
    "repeat",
    # Paramètres génératifs
    "S_mv",
    "D_mv",
    "E_mv",
    "P_mv",
    # Descripteurs émergents
    "D",
    "I",
    "V",
    "S",
    "E",
    "P",
    "BPM",
]