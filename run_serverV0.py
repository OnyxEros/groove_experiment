import os
import uvicorn
from pyngrok import ngrok
from dotenv import load_dotenv

from config import PORT

# =========================================================
# ENV
# =========================================================

load_dotenv("token.env")


# =========================================================
# NGROK
# =========================================================

def start_ngrok(port: int) -> str:
    token = os.getenv("NGROK_TOKEN")

    if not token:
        raise RuntimeError("NGROK_TOKEN missing (token.env or env var)")

    ngrok.set_auth_token(token)

    # IMPORTANT:
    # évite les tunnels fantômes lors des reloads
    ngrok.kill()

    url = ngrok.connect(port).public_url

    print("\n🌍 NGROK URL:")
    print(url)
    print()

    return url


# =========================================================
# SERVER
# =========================================================

def run_server():
    use_ngrok = os.getenv("USE_NGROK", "0") == "1"

    if use_ngrok:
        start_ngrok(PORT)

    uvicorn.run(
        "backend.api:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
        log_level="info",
    )


# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":
    run_server()