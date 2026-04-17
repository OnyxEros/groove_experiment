from backend.supabase_client import fetch_table
import pandas as pd


def load_dataset():
    data = fetch_table("responses")
    return pd.DataFrame(data)
