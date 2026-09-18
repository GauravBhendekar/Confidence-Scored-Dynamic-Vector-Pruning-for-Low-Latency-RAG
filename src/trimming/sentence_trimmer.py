import re
from typing import List
from src.common.types import Chunk, QueryTier, Granularity

class SentenceContextTrimmer:
    def __init__(self, sentence_limit: int = 2):
        self.sentence_limit = sentence_limit

    def trim_chunks(self, chunks: List[Chunk], query: str, tier: QueryTier) -> List[Chunk]:
        """
        Trims retained chunks based on query tier:
        - Tier 1: Extracts the top 1-2 sentence spans matching the query terms.
        - Tier 2/3: Preserves full chunk context without semantic truncation.
        """
        if tier != QueryTier.TIER_1_FACTOID:
            return chunks

        trimmed_chunks: List[Chunk] = []
        query_words = set(re.findall(r'\w+', query.lower()))

        for chunk in chunks:
            # If already a micro span, keep as is
            if chunk.granularity == Granularity.MICRO_SPAN:
                trimmed_chunks.append(chunk)
                continue

            # Segment into sentences
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', chunk.text) if s.strip()]
            if len(sentences) <= self.sentence_limit:
                trimmed_chunks.append(chunk)
                continue

            # Score sentences by query word overlap
            scored_sentences = []
            for s in sentences:
                s_words = set(re.findall(r'\w+', s.lower()))
                overlap = len(query_words.intersection(s_words))
                scored_sentences.append((overlap, s))

            # Select top N sentences preserving original text order
            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            top_sentences = set([s for _, s in scored_sentences[:self.sentence_limit]])

            selected_text = " ".join([s for s in sentences if s in top_sentences])
            token_cnt = len(selected_text.split())

            trimmed_chunk = chunk.model_copy(update={
                "text": selected_text,
                "token_count": token_cnt,
                "granularity": Granularity.MICRO_SPAN
            })
            trimmed_chunks.append(trimmed_chunk)

        return trimmed_chunks
