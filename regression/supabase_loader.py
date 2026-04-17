from infra.supabase_client import supabase
import pandas as pd


def load_responses():
    """
    Pull all responses from Supabase
    """
    res = supabase.table("responses").select("*").execute()

    if res.data is None:
        return pd.DataFrame()

    return pd.DataFrame(res.data)