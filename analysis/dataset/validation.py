from .schema import REQUIRED_COLUMNS


def validate_row(row):

    for col in REQUIRED_COLUMNS:
        if col not in row:
            raise ValueError(f"Missing field: {col}")