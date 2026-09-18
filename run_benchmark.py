import sys
import argparse
from src.common.config import ConfigManager
from src.common.logger import logger
from src.ingestion.markdown_loader import MarkdownLoader
from src.chunking.hierarchical_chunker import MultiGranularityChunker
from src.embeddings.bge_embedding import SentenceTransformerEmbeddingProvider
from src.indexing.vector_store import VectorStoreManager
from src.generation.llama_cpp_provider import LlamaCppProvider
from src.retrieval.baseline_pipeline import BaselineRAGPipeline
from src.retrieval.qatm_pipeline import QATMRAGPipeline
from src.evaluation.evaluator import BenchmarkEvaluator

def main():
    parser = argparse.ArgumentParser(description="Run Side-by-Side RAG Benchmark")
    parser.add_argument("--dataset", type=str, default="data/evaluation/eval_dataset.json", help="Path to evaluation dataset")
    parser.add_argument("--doc", type=str, default="ieee_project_dynamic_rag_architecture.md", help="Document to index")
    args = parser.parse_args()

    cfg = ConfigManager()
    logger.info("Setting up Benchmark Execution...")

    # Load & Index
    loader = MarkdownLoader()
    chunker = MultiGranularityChunker()
    embedder = SentenceTransformerEmbeddingProvider(model_name=cfg.get("embedding.model_name"), device="cpu")
    vector_store = VectorStoreManager(persist_directory="data/chroma_db", embedding_provider=embedder)
    llm = LlamaCppProvider(
        model_path=cfg.get("llm.model_path"),
        ollama_model=cfg.get("llm.ollama_model"),
        ollama_host=cfg.get("llm.ollama_host"),
        prefer_ollama=cfg.get("llm.prefer_ollama", False)
    )

    doc = loader.load(args.doc)
    chunks = chunker.chunk_document(doc)
    vector_store.clear()
    vector_store.add_chunks(chunks)

    # Initialize Pipelines
    baseline_pipe = BaselineRAGPipeline(vector_store, embedder, llm, top_k=5)
    qatm_pipe = QATMRAGPipeline(vector_store, embedder, llm)

    # Run Benchmark
    evaluator = BenchmarkEvaluator(baseline_pipe, qatm_pipe)
    report = evaluator.evaluate_dataset(args.dataset)

    summary = report["summary"]
    b_stats = summary["baseline_stats"]
    q_stats = summary["qatm_stats"]

    print("\n" + "="*70)
    print("                      BENCHMARK RESULTS SUMMARY                      ")
    print("="*70)
    print(f"Total Queries Evaluated: {summary['total_queries_evaluated']}")
    print("-" * 70)
    print(f"Metric                       | Baseline RAG      | QATM-RAG          ")
    print("-" * 70)
    print(f"Mean Context Tokens          | {b_stats['context_tokens']['mean']:<17} | {q_stats['context_tokens']['mean']:<17}")
    print(f"P95 Context Tokens           | {b_stats['context_tokens']['p95']:<17} | {q_stats['context_tokens']['p95']:<17}")
    print(f"Mean Token Savings (%)       | 0.0%              | {q_stats['token_savings_percent']['mean']}%")
    print(f"Mean Total Latency (ms)      | {b_stats['latency_ms']['mean']:<17} | {q_stats['latency_ms']['mean']:<17}")
    print(f"P95 Total Latency (ms)       | {b_stats['latency_ms']['p95']:<17} | {q_stats['latency_ms']['p95']:<17}")
    print(f"Avg Retained Chunks          | {b_stats['avg_retained_chunks']:<17} | {q_stats['avg_retained_chunks']:<17}")
    print("="*70)
    print("Report saved to 'results/benchmark_report.json'")

if __name__ == "__main__":
    main()
