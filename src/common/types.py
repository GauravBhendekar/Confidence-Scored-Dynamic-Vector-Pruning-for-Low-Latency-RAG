from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Granularity(str, Enum):
    MICRO_SPAN = "micro_span"       # Level 1: Sentence or small semantic unit
    STANDARD = "standard"           # Level 2: Paragraph chunk
    PARENT_BLOCK = "parent_block"   # Level 3: Full section or section group

class QueryTier(int, Enum):
    TIER_1_FACTOID = 1              # Factoid / Short Answer
    TIER_2_DESCRIPTIVE = 2          # Descriptive / Explanatory
    TIER_3_COMPLEX = 3              # Complex Synthesis

class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page_number: Optional[int] = None
    section_id: Optional[str] = None
    parent_block_id: Optional[str] = None
    standard_chunk_id: Optional[str] = None
    granularity: Granularity
    text: str
    token_count: int
    ordering_index: int
    source_location: Dict[str, int] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    similarity_score: Optional[float] = None

class Document(BaseModel):
    document_id: str
    document_name: str
    file_path: str
    file_type: str
    raw_text: str
    clean_text: str
    total_tokens: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ClassificationResult(BaseModel):
    query: str
    predicted_tier: QueryTier
    confidence: float
    reasoning: str
    is_fallback: bool = False

class PruningResult(BaseModel):
    initial_retrieved_count: int
    retained_count: int
    retained_chunks: List[Chunk]
    pruned_chunks: List[Chunk]
    knee_point_index: Optional[int] = None
    score_deltas: List[float] = Field(default_factory=list)
    strategy_used: str

class PipelineResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    query_tier: QueryTier
    classification_confidence: float
    retrieved_chunk_count: int
    retained_chunk_count: int
    context_tokens: int
    baseline_context_tokens: Optional[int] = None
    token_savings_percent: Optional[float] = None
    retrieval_latency_ms: float
    pruning_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    ttft_ms: Optional[float] = None
    trace_info: Dict[str, Any] = Field(default_factory=dict)
