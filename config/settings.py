"""
settings.py

- Loads API keys using python-dotenv
- Defines project constants (index name, chunk length fallback)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys (No OpenAI key needed for local Whisper)
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")  # Using GEMINI_API_KEY from .env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Pinecone configuration
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "video-embeddings")
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "768"))  # Default dimension for embeddings

# Project paths
BASE_DIR = Path(__file__).parent.parent
RAW_VIDEOS_DIR = BASE_DIR / "data" / "raw_videos"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHUNKS_DIR = BASE_DIR / "data" / "chunks"

# Ensure directories exist
RAW_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# Video processing parameters
DEFAULT_CHUNK_LENGTH = float(os.getenv("DEFAULT_CHUNK_LENGTH", "5.0"))  # seconds, fallback if no sentence boundaries
MAX_CHUNK_LENGTH = float(os.getenv("MAX_CHUNK_LENGTH", "15.0"))  # maximum allowed chunk length
MIN_CHUNK_LENGTH = float(os.getenv("MIN_CHUNK_LENGTH", "1.0"))  # minimum allowed chunk length

# Whisper configuration
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")  # Options: tiny, base, small, medium, large
LANGUAGE = os.getenv("LANGUAGE", "en")  # Default language

# Gemini configuration
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro-vision")
TEXT_EMBEDDING_MODEL = os.getenv("TEXT_EMBEDDING_MODEL", "embedding-001")  # Google's embedding model
