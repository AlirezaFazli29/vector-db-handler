import os
from dotenv import load_dotenv

# Load .env only if running locally
if os.getenv("ENV", "dev") == "dev":
    load_dotenv(".env")

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT"))
EMBEDDING_HOST = os.getenv("EMBEDDING_HOST")
EMBEDDING_PORT = int(os.getenv("EMBEDDING_PORT"))
