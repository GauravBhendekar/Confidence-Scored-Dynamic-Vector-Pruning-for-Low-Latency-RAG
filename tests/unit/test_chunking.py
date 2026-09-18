import pytest
from src.common.types import Document, Granularity
from src.chunking.hierarchical_chunker import MultiGranularityChunker

def test_hierarchical_chunking_lineage():
    doc = Document(
        document_id="doc_test_123",
        document_name="test.txt",
        file_path="/path/test.txt",
        file_type="txt",
        raw_text="Header text.\n\nParagraph one is descriptive text for standard chunking.\n\nParagraph two is another section.",
        clean_text="Header text.\n\nParagraph one is descriptive text for standard chunking.\n\nParagraph two is another section.",
        total_tokens=25
    )

    chunker = MultiGranularityChunker(parent_token_size=50, standard_token_size=20)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 0

    parent_chunks = [c for c in chunks if c.granularity == Granularity.PARENT_BLOCK]
    std_chunks = [c for c in chunks if c.granularity == Granularity.STANDARD]
    micro_chunks = [c for c in chunks if c.granularity == Granularity.MICRO_SPAN]

    assert len(parent_chunks) >= 1
    assert len(std_chunks) >= 1
    assert len(micro_chunks) >= 1

    # Verify bidirectional lineage linkage
    for m in micro_chunks:
        assert m.parent_block_id is not None
        assert m.standard_chunk_id is not None
