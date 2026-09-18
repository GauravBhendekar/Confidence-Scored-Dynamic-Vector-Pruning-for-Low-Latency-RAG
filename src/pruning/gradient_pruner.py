from typing import List
from src.common.types import Chunk, QueryTier, PruningResult
from src.common.logger import logger

class DynamicScoreGradientPruner:
    def __init__(
        self,
        gradient_threshold: float = 0.15,
        absolute_min_score: float = 0.50
    ):
        self.gradient_threshold = gradient_threshold
        self.absolute_min_score = absolute_min_score

    def prune(self, chunks: List[Chunk], tier: QueryTier) -> PruningResult:
        if not chunks:
            return PruningResult(
                initial_retrieved_count=0,
                retained_count=0,
                retained_chunks=[],
                pruned_chunks=[],
                knee_point_index=None,
                score_deltas=[],
                strategy_used="empty_input"
            )

        # Sort chunks descending by similarity score
        sorted_chunks = sorted(chunks, key=lambda c: c.similarity_score or 0.0, reverse=True)
        scores = [c.similarity_score or 0.0 for c in sorted_chunks]

        # Calculate score deltas
        deltas = []
        for i in range(len(scores) - 1):
            delta = round(scores[i] - scores[i + 1], 4)
            deltas.append(delta)

        # Determine Tier Min/Max bounds.
        # Tier 3's max is bounded by the baseline's fixed k (5) so QATM never
        # retains more chunks than the baseline it is compared against; extra
        # depth for complex queries comes from a wider candidate pool, not a
        # larger final context.
        if tier == QueryTier.TIER_1_FACTOID:
            min_k, max_k = 1, 2
        elif tier == QueryTier.TIER_2_DESCRIPTIVE:
            min_k, max_k = 2, 4
        else:  # Tier 3
            min_k, max_k = 3, 5

        # Find knee-point (maximum score delta)
        knee_index = None
        if deltas:
            knee_index = deltas.index(max(deltas))

        # Dynamic Cutoff Logic
        cutoff_k = len(sorted_chunks)

        # 1. Absolute Score Gate
        for idx, score in enumerate(scores):
            if score < self.absolute_min_score:
                cutoff_k = idx
                break

        # 2. Gradient / Information Cliff Cutoff
        for idx, delta in enumerate(deltas):
            if delta >= self.gradient_threshold and idx + 1 < cutoff_k:
                cutoff_k = idx + 1
                break

        # 2b. Flat-distribution handling for complex queries.
        # A flat score curve (max delta well below the cliff threshold) means the
        # evidence is spread evenly across many chunks rather than concentrated
        # behind one knee. For Tier 3 this is the multi-hop case, so clamping to
        # min_k would silently drop needed facts. Widen to max_k instead.
        is_flat = bool(deltas) and max(deltas) < self.gradient_threshold
        if tier == QueryTier.TIER_3_COMPLEX and is_flat:
            cutoff_k = max(cutoff_k, max_k)

        # 3. Enforce Tier Constraints (min_k <= final_k <= max_k)
        final_k = max(min_k, min(cutoff_k, max_k, len(sorted_chunks)))

        retained = sorted_chunks[:final_k]
        pruned = sorted_chunks[final_k:]

        logger.info(
            f"[Pruner] Retained {len(retained)}/{len(sorted_chunks)} chunks for Tier {tier.value}. "
            f"Knee index: {knee_index}, Max Delta: {max(deltas) if deltas else 0.0}, Flat: {is_flat}"
        )

        return PruningResult(
            initial_retrieved_count=len(sorted_chunks),
            retained_count=len(retained),
            retained_chunks=retained,
            pruned_chunks=pruned,
            knee_point_index=knee_index,
            score_deltas=deltas,
            strategy_used="flat_widened" if (tier == QueryTier.TIER_3_COMPLEX and is_flat) else "gradient_knee_tier_constrained"
        )
