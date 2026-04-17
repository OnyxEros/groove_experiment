from infra.supabase_client import fetch_responses
import pandas as pd


def load_dataset():
    data = fetch_responses()
    return pd.DataFrame(data)