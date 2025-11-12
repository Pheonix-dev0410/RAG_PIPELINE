"""
Enhanced Configuration for Video RAG Pipeline with Transcript Support
Addresses: Deprecated imports, optimal chunking, better embedding model
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Dict, Any
import os

# ============================================================================
# EMBEDDING MODEL - Upgraded for better retrieval quality
# ============================================================================
# Using better embedding model with 768 dimensions (vs 384 in old model)
EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",  # Better than all-MiniLM-L6-v2
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}  # L2 normalization for cosine similarity
)

# ============================================================================
# CHUNKING STRATEGY - Optimized for video metadata + transcripts
# ============================================================================

# For VIDEO METADATA (links, titles, descriptions) - smaller chunks
VIDEO_METADATA_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=800,  # Increased from 500
    chunk_overlap=150,  # 20% overlap (was 10%)
    separators=["\n\n", "\n", ".", "!", "?", " ", ""],  # Respect structure
    length_function=len,
)

# For VIDEO TRANSCRIPTS (long conversational content) - larger chunks
TRANSCRIPT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1200,  # Larger for contextual answers
    chunk_overlap=300,  # 25% overlap for continuity
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    length_function=len,
)

# Default splitter (used when type is unknown)
TEXT_SPLITTER = VIDEO_METADATA_SPLITTER

# ============================================================================
# VECTOR STORE CONFIGURATION
# ============================================================================
FAISS_DIR = "faiss_index"
FAISS_BACKUP_DIR = "faiss_index_backup"

# Hybrid search configuration
USE_HYBRID_SEARCH = True  # Set to False to use pure FAISS
HYBRID_DENSE_WEIGHT = 0.5  # Weight for vector search (0.0 to 1.0)
HYBRID_SPARSE_WEIGHT = 0.5  # Weight for keyword search (0.0 to 1.0)

# ============================================================================
# RETRIEVAL CONFIGURATION - Optimized for dual-mode operation
# ============================================================================

# For VIDEO LINK RETRIEVAL (when user asks for a video)
VIDEO_RETRIEVAL_CONFIG: Dict[str, Any] = {
    "search_type": "similarity_score_threshold",  # Only get relevant videos
    "search_kwargs": {
        "k": 5,  # Reduced from 10
        "score_threshold": 0.6,  # Only fetch if similarity > 0.6
        "filter": {"content_type": "video_metadata"}  # Only search video metadata
    }
}

# For TRANSCRIPT Q&A (when user asks about watched video)
TRANSCRIPT_RETRIEVAL_CONFIG: Dict[str, Any] = {
    "search_type": "mmr",  # Diversity for conversational responses
    "search_kwargs": {
        "k": 4,  # Focused context
        "fetch_k": 10,  # Reduced from 20
        "lambda_mult": 0.8,  # More relevance, less diversity
        "filter": {"content_type": "transcript"}  # Only search transcripts
    }
}

# ============================================================================
# LLM CONFIGURATION
# ============================================================================
LLM_CONFIG: Dict[str, Any] = {
    "model": "gpt-4o",
    "temperature": 0.7,  # Higher for natural, conversational responses
    "max_tokens": 800,  # Increased from 512 for detailed answers
    "request_timeout": 30,  # 30 second timeout
    "max_retries": 3,  # Retry on failures
}

# For video link retrieval (factual, deterministic)
LLM_CONFIG_VIDEO_RETRIEVAL: Dict[str, Any] = {
    "model": "gpt-4o",
    "temperature": 0.2,  # Lower for precise link retrieval
    "max_tokens": 300,
    "request_timeout": 20,
    "max_retries": 3,
}

# For transcript Q&A (conversational, engaging)
LLM_CONFIG_TRANSCRIPT_QA: Dict[str, Any] = {
    "model": "gpt-4o",
    "temperature": 0.7,  # Higher for human-like responses
    "max_tokens": 800,
    "request_timeout": 30,
    "max_retries": 3,
}

# ============================================================================
# MEMORY CONFIGURATION - Per-user, not global
# ============================================================================
MEMORY_CONFIG: Dict[str, Any] = {
    "memory_type": "buffer_window",  # Not summary (too slow)
    "k": 5,  # Last 5 interactions
    "return_messages": True,
}

# ============================================================================
# SESSION CONFIGURATION
# ============================================================================
SESSION_CONFIG: Dict[str, Any] = {
    "session_timeout": 3600,  # 1 hour
    "max_sessions": 1000,  # Max concurrent sessions
    "cleanup_interval": 300,  # Cleanup every 5 minutes
}

# ============================================================================
# METADATA TAGS for content type identification
# ============================================================================
CONTENT_TYPE_VIDEO = "video_metadata"
CONTENT_TYPE_TRANSCRIPT = "transcript"

# ============================================================================
# VALIDATION
# ============================================================================
def validate_config() -> bool:
    """Validate configuration on import"""
    required_env_vars = ["OPENAI_API_KEY"]  # Standard OpenAI key name
    missing = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing:
        raise ValueError(f"Missing required environment variables: {missing}")
    
    # Ensure directories exist
    os.makedirs(FAISS_DIR, exist_ok=True)
    os.makedirs(FAISS_BACKUP_DIR, exist_ok=True)
    
    return True

