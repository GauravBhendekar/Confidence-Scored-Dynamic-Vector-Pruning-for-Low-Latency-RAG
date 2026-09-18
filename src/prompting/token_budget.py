from typing import List
from src.common.types import Chunk, QueryTier
from src.common.logger import logger

class TokenBudgetController:
    def __init__(self):
        # Configurable token caps per tier. These are hard ceilings on the
        # context handed to the LLM, independent of how many chunks were retained.
        self.tier_caps = {
            QueryTier.TIER_1_FACTOID: 200,
            QueryTier.TIER_2_DESCRIPTIVE: 700,
            QueryTier.TIER_3_COMPLEX: 1200
        }

    def enforce_budget(self, chunks: List[Chunk], tier: QueryTier) -> List[Chunk]:
        max_budget = self.tier_caps.get(tier, 700)
        selected_chunks: List[Chunk] = []
        current_tokens = 0

        for chunk in chunks:
            # Recalculate token count if missing or invalid
            token_cnt = chunk.token_count if chunk.token_count and chunk.token_count > 0 else len(chunk.text.split())

            if current_tokens + token_cnt <= max_budget:
                selected_chunks.append(chunk)
                current_tokens += token_cnt
            else:
                remaining_budget = max_budget - current_tokens
                if remaining_budget > 20:
                    words = chunk.text.split()[:remaining_budget]
                    if words:
                        trimmed_text = " ".join(words)
                        trimmed_chunk = chunk.model_copy(update={
                            "text": trimmed_text,
                            "token_count": len(words)
                        })
                        selected_chunks.append(trimmed_chunk)
                        current_tokens += len(words)
                logger.info(f"[TokenBudget] Cap reached ({current_tokens}/{max_budget} tokens). Budget enforced.")
                break

        return selected_chunks

