"""
Embedding Provider Module.
"""
from .base_embedding import EmbeddingProvider
from .bge_embedding import SentenceTransformerEmbeddingProvider

__all__ = ["EmbeddingProvider", "SentenceTransformerEmbeddingProvider"]
