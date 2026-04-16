from dotenv import load_dotenv
from pathlib import Path

# force chargement depuis la racine du projet
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

import uvicorn
from config import PORT


def run_server():
    uvicorn.run(
        "backend.api:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()