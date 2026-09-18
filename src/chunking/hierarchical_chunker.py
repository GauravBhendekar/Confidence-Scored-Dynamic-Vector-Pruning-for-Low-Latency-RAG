import re
import uuid
from typing import List, Dict, Tuple
from src.common.types import Document, Chunk, Granularity

class MultiGranularityChunker:
    def __init__(
        self,
        parent_token_size: int = 700,
        standard_token_size: int = 200,
        micro_span_sentence_limit: int = 1
    ):
        self.parent_token_size = parent_token_size
        self.standard_token_size = standard_token_size
        self.micro_span_sentence_limit = micro_span_sentence_limit

    def chunk_document(self, doc: Document) -> List[Chunk]:
        """
        Splits a Document into Level 3 (Parent Blocks), Level 2 (Standard Chunks),
        and Level 1 (Micro-Spans) with bidirectional lineage metadata.
        """
        all_chunks: List[Chunk] = []
        raw_paragraphs = [p.strip() for p in doc.clean_text.split("\n\n") if p.strip()]

        if not raw_paragraphs:
            raw_paragraphs = [doc.clean_text] if doc.clean_text.strip() else []

        global_ordering_idx = 0
        parent_blocks = self._assemble_parent_blocks(raw_paragraphs)

        for p_idx, (p_text, p_paragraphs) in enumerate(parent_blocks):
            parent_id = f"parent_{doc.document_id}_{p_idx}_{uuid.uuid4().hex[:6]}"
            p_tokens = len(p_text.split())

            # Level 3: Parent Block Chunk
            parent_chunk = Chunk(
                chunk_id=parent_id,
                document_id=doc.document_id,
                document_name=doc.document_name,
                parent_block_id=parent_id,
                standard_chunk_id=None,
                granularity=Granularity.PARENT_BLOCK,
                text=p_text,
                token_count=p_tokens,
                ordering_index=global_ordering_idx
            )
            global_ordering_idx += 1
            all_chunks.append(parent_chunk)

            # Level 2: Standard Chunks (Paragraphs within Parent Block)
            for s_idx, para in enumerate(p_paragraphs):
                std_id = f"std_{parent_id}_{s_idx}_{uuid.uuid4().hex[:6]}"
                s_tokens = len(para.split())

                std_chunk = Chunk(
                    chunk_id=std_id,
                    document_id=doc.document_id,
                    document_name=doc.document_name,
                    parent_block_id=parent_id,
                    standard_chunk_id=std_id,
                    granularity=Granularity.STANDARD,
                    text=para,
                    token_count=s_tokens,
                    ordering_index=global_ordering_idx
                )
                global_ordering_idx += 1
                all_chunks.append(std_chunk)

                # Level 1: Micro-Spans (Sentences within Standard Chunk)
                sentences = self._split_sentences(para)
                for m_idx, sentence in enumerate(sentences):
                    micro_id = f"micro_{std_id}_{m_idx}_{uuid.uuid4().hex[:6]}"
                    m_tokens = len(sentence.split())

                    micro_chunk = Chunk(
                        chunk_id=micro_id,
                        document_id=doc.document_id,
                        document_name=doc.document_name,
                        parent_block_id=parent_id,
                        standard_chunk_id=std_id,
                        granularity=Granularity.MICRO_SPAN,
                        text=sentence,
                        token_count=m_tokens,
                        ordering_index=global_ordering_idx
                    )
                    global_ordering_idx += 1
                    all_chunks.append(micro_chunk)

        return all_chunks

    def _assemble_parent_blocks(self, paragraphs: List[str]) -> List[Tuple[str, List[str]]]:
        """Groups paragraphs into parent blocks targetting parent_token_size."""
        blocks: List[Tuple[str, List[str]]] = []
        current_paras: List[str] = []
        current_count = 0

        for para in paragraphs:
            para_count = len(para.split())
            if current_count + para_count > self.parent_token_size and current_paras:
                block_text = "\n\n".join(current_paras)
                blocks.append((block_text, current_paras.copy()))
                current_paras = [para]
                current_count = para_count
            else:
                current_paras.append(para)
                current_count += para_count

        if current_paras:
            block_text = "\n\n".join(current_paras)
            blocks.append((block_text, current_paras))

        return blocks

    def _split_sentences(self, text: str) -> List[str]:
        """Splits paragraph text into clean sentence spans."""
        # Simple regex sentence splitting
        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences if sentences else [text]
