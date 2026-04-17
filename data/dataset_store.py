import pandas as pd
from config import RESP_FILE
from backend.supabase_client import fetch_table


def sync_responses():
    """
    Pull Supabase → local CSV snapshot
    """

    data = fetch_table("responses")

    df = pd.DataFrame(data)

    RESP_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESP_FILE, index=False)

    return df


def load_responses():
    """
    Load ML-safe dataset (local only)
    """

    if not RESP_FILE.exists():
        raise FileNotFoundError(
            "No local dataset found. Run sync_responses() first."
        )

    return pd.read_csv(RESP_FILE)
