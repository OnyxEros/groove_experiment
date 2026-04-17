from infra.supabase_client import get_supabase


def fetch_ratings(table="groove_ratings"):
    """
    Load human ratings from Supabase.
    Expected schema:
        - stimulus_id
        - rating
    """

    supabase = get_supabase()

    res = supabase.table(table).select("*").execute()

    if not res.data:
        raise ValueError("No ratings found in Supabase")

    return res.data
