# config.py
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# =========================
# Database (MySQL / MariaDB)
# =========================
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_DATABASE = os.getenv("DB_DATABASE", "myvault")

# =========================
# Redis Configuration
# =========================
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_BRPOP_TIMEOUT = int(os.getenv("REDIS_BRPOP_TIMEOUT", 5))

# =========================
# Queues
# =========================
INPUT_QUEUE = os.getenv(
    "INPUT_QUEUE",
    "fliqz_moderation_image_video_queue"
)

# =========================
# Local LLaMA / Ollama
# =========================
LLAMA_API_URL = os.getenv(
    "LLAMA_API_URL",
    "http://localhost:11434/api/generate"
)
