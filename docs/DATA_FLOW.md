# QATM-RAG Data Flow Specification

## 1. Document Ingestion Flow
`Raw PDF/TXT/MD Document` $\rightarrow$ `DocumentLoader` $\rightarrow$ `TextCleaner` $\rightarrow$ `MultiGranularityChunker` $\rightarrow$ `Level 1 / 2 / 3 Chunks` $\rightarrow$ `SentenceTransformerEmbeddingProvider` $\rightarrow$ `VectorStoreManager (ChromaDB)`

## 2. Online Retrieval & Pruning Flow
`User Question` $\rightarrow$ `QueryComplexityClassifier` $\rightarrow$ `Tier 1 / 2 / 3` $\rightarrow$ `Vector Search (N=15, tier granularity)` $\rightarrow$ `Text-Dedup` $\rightarrow$ `DynamicScoreGradientPruner` $\rightarrow$ `SentenceContextTrimmer` $\rightarrow$ `TokenBudgetController` $\rightarrow$ `MinimalPromptAssembler` $\rightarrow$ `LLMProvider (GGUF / Ollama / fallback)` $\rightarrow$ `Answer + Sources + Metrics`

## 3. Benchmark Flow
`BenchmarkEvaluator` $\rightarrow$ `LLM warm_up()` $\rightarrow$ per query: `BaselineRAGPipeline (k=5)` and `QATMRAGPipeline` $\rightarrow$ `MetricsAggregator` $\rightarrow$ `results/benchmark_report.json`

The report records `summary.llm_status` (`backend`, `is_real_llm`,
`metrics_valid`, `warmed_up`). If `metrics_valid` is `false`, the extractive
fallback ran and latency/quality numbers must not be cited.

