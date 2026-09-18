import time
from typing import Dict, Any, Optional
from src.indexing.vector_store import VectorStoreManager
from src.embeddings.base_embedding import EmbeddingProvider
from src.generation.base_llm import LLMProvider
from src.classification.classifier import QueryComplexityClassifier
from src.pruning.gradient_pruner import DynamicScoreGradientPruner
from src.trimming.sentence_trimmer import SentenceContextTrimmer
from src.prompting.token_budget import TokenBudgetController
from src.prompting.prompt_builder import MinimalPromptAssembler
from src.common.types import PipelineResponse, QueryTier, Granularity
from src.common.logger import logger

class QATMRAGPipeline:
    def __init__(
        self,
        vector_store: VectorStoreManager,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
        classifier: Optional[QueryComplexityClassifier] = None,
        pruner: Optional[DynamicScoreGradientPruner] = None,
        trimmer: Optional[SentenceContextTrimmer] = None,
        budget_controller: Optional[TokenBudgetController] = None,
        candidate_pool_size: int = 15
    ):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider
        self.classifier = classifier or QueryComplexityClassifier()
        self.pruner = pruner or DynamicScoreGradientPruner()
        self.trimmer = trimmer or SentenceContextTrimmer()
        self.budget_controller = budget_controller or TokenBudgetController()
        self.candidate_pool_size = candidate_pool_size

    @staticmethod
    def _dedupe_by_text(chunks):
        """Drop chunks whose normalized text duplicates one already kept.

        Chunks arrive sorted by descending similarity, so the first occurrence
        (highest score) is retained and later duplicates are discarded.
        """
        seen = set()
        unique = []
        for chunk in chunks:
            key = " ".join(chunk.text.lower().split())
            if key in seen:
                continue
            seen.add(key)
            unique.append(chunk)
        return unique

    def run(self, query: str, baseline_tokens: Optional[int] = None) -> PipelineResponse:
        start_total = time.time()

        # Step 1: Query Complexity Classification
        class_res = self.classifier.classify(query)
        tier = class_res.predicted_tier

        # Map Query Tier to Retrieval Granularity.
        # Note: Tier 3 uses STANDARD chunks (not parent blocks). Parent blocks
        # (~700 tokens each) would make complex queries retrieve MORE tokens than
        # the fixed-k baseline, inverting the token-minimization objective. Complex
        # queries instead get a WIDER pool of standard chunks via a raised k cap.
        if tier == QueryTier.TIER_1_FACTOID:
            granularity = Granularity.MICRO_SPAN
        else:
            granularity = Granularity.STANDARD

        # Step 2: Vector Candidate Search
        start_ret = time.time()
        query_vector = self.embedding_provider.embed_text(query)
        candidate_chunks = self.vector_store.search(
            query_vector=query_vector,
            top_k=self.candidate_pool_size,
            granularity_filter=granularity
        )
        
        # Fallback if specific granularity returns empty
        if not candidate_chunks:
            candidate_chunks = self.vector_store.search(
                query_vector=query_vector,
                top_k=self.candidate_pool_size,
                granularity_filter=None
            )

        # Deduplicate near-identical text. Because every paragraph is indexed at
        # three granularities, the same sentence can appear as a standard chunk
        # and as a micro-span, so without this the candidate pool wastes budget
        # on duplicates instead of covering distinct evidence.
        candidate_chunks = self._dedupe_by_text(candidate_chunks)
        ret_latency = round((time.time() - start_ret) * 1000, 2)

        # Step 3: Dynamic Score Gradient Pruning
        start_prune = time.time()
        prune_res = self.pruner.prune(candidate_chunks, tier)
        retained_chunks = prune_res.retained_chunks

        # Step 4: Sentence Context Trimming
        trimmed_chunks = self.trimmer.trim_chunks(retained_chunks, query, tier)

        # Step 5: Token Budget Enforcement
        final_chunks = self.budget_controller.enforce_budget(trimmed_chunks, tier)
        prune_latency = round((time.time() - start_prune) * 1000, 2)

        # Step 6: Minimal Prompt Building
        system_prompt, user_prompt, stats = MinimalPromptAssembler.build_prompt(query, final_chunks)
        current_context_tokens = stats["context_tokens"]

        # Step 7: LLM Generation
        gen_res = self.llm_provider.generate(user_prompt, system_prompt=system_prompt)
        gen_latency = gen_res["latency_ms"]

        total_latency = round((time.time() - start_total) * 1000, 2)

        # Calculate Token Savings vs Baseline
        token_savings_pct = 0.0
        if baseline_tokens and baseline_tokens > 0:
            token_savings_pct = round(((baseline_tokens - current_context_tokens) / baseline_tokens) * 100, 2)

        sources = [
            {"chunk_id": c.chunk_id, "document_name": c.document_name, "score": c.similarity_score, "granularity": c.granularity.value}
            for c in final_chunks
        ]

        return PipelineResponse(
            query=query,
            answer=gen_res["text"],
            sources=sources,
            query_tier=tier,
            classification_confidence=class_res.confidence,
            retrieved_chunk_count=len(candidate_chunks),
            retained_chunk_count=len(final_chunks),
            context_tokens=current_context_tokens,
            baseline_context_tokens=baseline_tokens,
            token_savings_percent=token_savings_pct,
            retrieval_latency_ms=ret_latency,
            pruning_latency_ms=prune_latency,
            generation_latency_ms=gen_latency,
            total_latency_ms=total_latency,
            ttft_ms=gen_res.get("ttft_ms"),
            trace_info={
                "pipeline": "qatm_rag",
                "tier_name": tier.name,
                "reasoning": class_res.reasoning,
                "knee_point_index": prune_res.knee_point_index,
                "deltas": prune_res.score_deltas
            }
        )
