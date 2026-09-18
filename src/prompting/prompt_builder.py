from typing import List, Tuple, Dict, Any
from src.common.types import Chunk

class MinimalPromptAssembler:
    @staticmethod
    def build_prompt(query: str, chunks: List[Chunk]) -> Tuple[str, str, Dict[str, int]]:
        """
        Assembles a minimal prompt with clear context separation and metadata tracing.
        Returns: (system_prompt, user_prompt, token_stats)
        """
        system_prompt = (
            "You are a precise AI assistant. Answer the user's question using ONLY the provided context below. "
            "If the answer cannot be found in the context, state 'Information not found in context.' "
            "Do not hallucinate or make claims outside the context."
        )

        context_blocks = []
        context_tokens = 0

        for i, chunk in enumerate(chunks):
            doc_info = f"[{chunk.document_name} | Section: {chunk.parent_block_id or 'General'}]"
            block = f"Source {i+1} {doc_info}:\n{chunk.text.strip()}"
            context_blocks.append(block)
            context_tokens += chunk.token_count

        formatted_context = "\n\n".join(context_blocks)
        user_prompt = f"CONTEXT:\n{formatted_context}\n\nQUESTION: {query}\n\nANSWER:"

        system_tokens = len(system_prompt.split())
        question_tokens = len(query.split())
        total_input_tokens = system_tokens + context_tokens + question_tokens

        stats = {
            "system_tokens": system_tokens,
            "context_tokens": context_tokens,
            "question_tokens": question_tokens,
            "total_input_tokens": total_input_tokens
        }

        return system_prompt, user_prompt, stats
