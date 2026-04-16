from supabase import create_client
import os

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

print("URL:", SUPABASE_URL)
print("KEY OK:", bool(SUPABASE_KEY))
print("KEY HEAD:", (SUPABASE_KEY or "")[:10])
print("KEY RAW:", repr(os.getenv("SUPABASE_KEY")))
print("KEY LEN:", len(os.getenv("SUPABASE_KEY") or ""))

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)