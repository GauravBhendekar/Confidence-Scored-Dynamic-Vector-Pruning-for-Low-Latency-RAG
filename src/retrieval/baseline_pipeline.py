import time
from typing import Dict, Any
from src.indexing.vector_store import VectorStoreManager
from src.embeddings.base_embedding import EmbeddingProvider
from src.generation.base_llm import LLMProvider
from src.prompting.prompt_builder import MinimalPromptAssembler
from src.common.types import PipelineResponse, QueryTier, Granularity

class BaselineRAGPipeline:
    def __init__(
        self,
        vector_store: VectorStoreManager,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
        top_k: int = 5
    ):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider
        self.top_k = top_k

    def run(self, query: str) -> PipelineResponse:
        start_total = time.time()

        # Step 1: Embed Query
        start_ret = time.time()
        query_vector = self.embedding_provider.embed_text(query)

        # Step 2: Top-K Vector Retrieval (Standard Granularity)
        chunks = self.vector_store.search(
            query_vector=query_vector,
            top_k=self.top_k,
            granularity_filter=Granularity.STANDARD
        )
        ret_latency = round((time.time() - start_ret) * 1000, 2)

        # Step 3: Minimal Prompt Assembly
        system_prompt, user_prompt, stats = MinimalPromptAssembler.build_prompt(query, chunks)

        # Step 4: Generation
        gen_res = self.llm_provider.generate(user_prompt, system_prompt=system_prompt)
        gen_latency = gen_res["latency_ms"]

        total_latency = round((time.time() - start_total) * 1000, 2)

        sources = [
            {"chunk_id": c.chunk_id, "document_name": c.document_name, "score": c.similarity_score}
            for c in chunks
        ]

        return PipelineResponse(
            query=query,
            answer=gen_res["text"],
            sources=sources,
            query_tier=QueryTier.TIER_2_DESCRIPTIVE,
            classification_confidence=1.0,
            retrieved_chunk_count=len(chunks),
            retained_chunk_count=len(chunks),
            context_tokens=stats["context_tokens"],
            baseline_context_tokens=stats["context_tokens"],
            token_savings_percent=0.0,
            retrieval_latency_ms=ret_latency,
            pruning_latency_ms=0.0,
            generation_latency_ms=gen_latency,
            total_latency_ms=total_latency,
            ttft_ms=gen_res.get("ttft_ms"),
            trace_info={"pipeline": "baseline_rag", "top_k": self.top_k}
        )
