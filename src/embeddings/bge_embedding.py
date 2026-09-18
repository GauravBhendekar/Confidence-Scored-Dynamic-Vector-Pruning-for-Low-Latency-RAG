from typing import List
from sentence_transformers import SentenceTransformer
from src.embeddings.base_embedding import EmbeddingProvider
from src.common.logger import logger

class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        logger.info(f"Loading Embedding Model '{model_name}' on device '{device}'...")
        self.model = SentenceTransformer(model_name, device=device)
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = self.model.get_embedding_dimension()
        else:
            self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> List[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(texts, batch_size=32, normalize_embeddings=True)
        return embeddings.tolist()

    def get_dimension(self) -> int:
        return self.dimension
