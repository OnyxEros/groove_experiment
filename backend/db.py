from supabase import create_client
from config import SUPABASE_URL, SUPABASE_KEY

# client Supabase global (réutilisé par toute l'app)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
