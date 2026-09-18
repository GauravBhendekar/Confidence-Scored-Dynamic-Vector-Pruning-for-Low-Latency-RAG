import os
import chromadb
from typing import List, Dict, Any, Optional
from src.common.types import Chunk, Granularity
from src.embeddings.base_embedding import EmbeddingProvider
from src.common.logger import logger

class VectorStoreManager:
    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "qatm_documents",
        embedding_provider: Optional[EmbeddingProvider] = None
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_provider = embedding_provider
        
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def _ensure_collection(self):
        try:
            self.collection.count()
        except Exception:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

    def add_chunks(self, chunks: List[Chunk]):
        if not chunks:
            return

        self._ensure_collection()
        ids = []
        documents = []
        embeddings = []
        metadatas = []

        texts = [chunk.text for chunk in chunks]
        if self.embedding_provider:
            embeddings_list = self.embedding_provider.embed_documents(texts)
        else:
            embeddings_list = [chunk.embedding for chunk in chunks if chunk.embedding is not None]

        for i, chunk in enumerate(chunks):
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            embeddings.append(embeddings_list[i])
            metadatas.append({
                "document_id": chunk.document_id,
                "document_name": chunk.document_name,
                "parent_block_id": chunk.parent_block_id or "",
                "standard_chunk_id": chunk.standard_chunk_id or "",
                "granularity": chunk.granularity.value,
                "token_count": chunk.token_count,
                "ordering_index": chunk.ordering_index
            })

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        logger.info(f"Successfully indexed {len(chunks)} chunks into ChromaDB collection '{self.collection_name}'.")

    def search(
        self,
        query_vector: List[float],
        top_k: int = 15,
        granularity_filter: Optional[Granularity] = None
    ) -> List[Chunk]:
        self._ensure_collection()
        where_filter = None
        if granularity_filter:
            where_filter = {"granularity": granularity_filter.value}

        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            logger.warning(f"Collection query error ({e}). Refreshing collection.")
            self._ensure_collection()
            try:
                results = self.collection.query(
                    query_embeddings=[query_vector],
                    n_results=top_k,
                    where=where_filter,
                    include=["documents", "metadatas", "distances"]
                )
            except Exception:
                return []

        chunks: List[Chunk] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return chunks

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i in range(len(ids)):
            # Convert cosine distance to cosine similarity (1.0 - distance)
            sim_score = round(1.0 - distances[i], 4)
            meta = metadatas[i]

            chunks.append(
                Chunk(
                    chunk_id=ids[i],
                    document_id=meta["document_id"],
                    document_name=meta["document_name"],
                    parent_block_id=meta.get("parent_block_id") or None,
                    standard_chunk_id=meta.get("standard_chunk_id") or None,
                    granularity=Granularity(meta["granularity"]),
                    text=documents[i],
                    token_count=meta["token_count"],
                    ordering_index=meta["ordering_index"],
                    similarity_score=sim_score
                )
            )

        return chunks

    def clear(self):
        try:
            self._ensure_collection()
            all_ids = self.collection.get()["ids"]
            if all_ids:
                self.collection.delete(ids=all_ids)
            logger.info("ChromaDB collection cleared.")
        except Exception as e:
            logger.warning(f"Error clearing collection via delete IDs ({e}). Recreating collection...")
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

    def get_count(self) -> int:
        try:
            self._ensure_collection()
            return self.collection.count()
        except Exception:
            return 0


