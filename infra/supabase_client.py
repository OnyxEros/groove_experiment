import os
from supabase import create_client, Client


# =========================================================
# CLIENT INITIALIZATION
# =========================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase credentials in environment 
variables")


client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# =========================================================
# GENERIC WRAPPERS
# =========================================================

def fetch_table(table_name: str, select: str = "*"):
    return client.table(table_name).select(select).execute().data


def insert_rows(table_name: str, rows: list[dict]):
    return client.table(table_name).insert(rows).execute()


def upsert_rows(table_name: str, rows: list[dict]):
    return client.table(table_name).upsert(rows).execute()


def query(table_name: str, filters: dict):
    q = client.table(table_name).select("*")

    for k, v in filters.items():
        q = q.eq(k, v)

    return q.execute().data
