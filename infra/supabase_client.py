import os
from supabase import create_client, Client

_client: Client | None = None

def get_supabase() -> Client:
    global _client
    if _client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError("Missing env vars: SUPABASE_URL / SUPABASE_KEY")
        _client = create_client(url, key)
    return _client


# ✅ ALIAS (fix immédiat de ton bug)
def get_supabase_client() -> Client:
    return get_supabase()


def insert_response(row: dict):
    return get_supabase().table("responses").insert(row).execute()


def fetch_responses() -> list[dict]:
    return get_supabase().table("responses").select("*").execute().data