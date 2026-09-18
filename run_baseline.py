import sys
import json
import argparse
from src.common.config import ConfigManager
from src.common.logger import logger
from src.ingestion.markdown_loader import MarkdownLoader
from src.chunking.hierarchical_chunker import MultiGranularityChunker
from src.embeddings.bge_embedding import SentenceTransformerEmbeddingProvider
from src.indexing.vector_store import VectorStoreManager
from src.generation.llama_cpp_provider import LlamaCppProvider
from src.retrieval.baseline_pipeline import BaselineRAGPipeline

def main():
    parser = argparse.ArgumentParser(description="Run Baseline RAG Pipeline")
    parser.add_argument("--dataset", type=str, default="data/evaluation/eval_dataset.json", help="Path to evaluation dataset")
    parser.add_argument("--doc", type=str, default="ieee_project_dynamic_rag_architecture.md", help="Sample document to index")
    args = parser.parse_args()

    cfg = ConfigManager()
    
    # 1. Initialize Components
    logger.info("Initializing Baseline RAG Pipeline...")
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

    # 2. Ingest & Index Document
    doc = loader.load(args.doc)
    chunks = chunker.chunk_document(doc)
    vector_store.clear()
    vector_store.add_chunks(chunks)

    # 3. Initialize Pipeline
    pipeline = BaselineRAGPipeline(vector_store, embedder, llm, top_k=5)

    # 4. Load dataset and run
    with open(args.dataset, "r", encoding="utf-8") as f:
        queries = json.load(f)

    logger.info(f"Running Baseline RAG on {len(queries)} queries...")
    for q_item in queries:
        res = pipeline.run(q_item["question"])
        print("\n" + "="*50)
        print(f"QUERY: {res.query}")
        print(f"BASELINE CONTEXT TOKENS: {res.context_tokens}")
        print(f"LATENCY: {res.total_latency_ms} ms")
        print(f"ANSWER: {res.answer[:120]}...")

if __name__ == "__main__":
    main()
