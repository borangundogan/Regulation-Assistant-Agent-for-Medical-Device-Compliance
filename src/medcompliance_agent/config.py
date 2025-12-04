from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
REGULATIONS_DIR = DATA_DIR / "regulations"

# Chunking configuration
CHUNK_SIZE_CHARS = 800
CHUNK_OVERLAP_CHARS = 200

# Retrieval configuration
TOP_K_BM25 = 10
TOP_K_DENSE = 10
TOP_K_HYBRID = 5

ALPHA_DENSE = 0.5
BETA_SPARSE = 0.5

# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
