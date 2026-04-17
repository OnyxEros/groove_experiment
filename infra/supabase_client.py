import os
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing Supabase env vars")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# =========================================================
# API CLEAN WRAPPERS
# =========================================================

def insert_response(row: dict):
    return supabase.table("responses").insert(row).execute()


def fetch_responses():
    return supabase.table("responses").select("*").execute().data