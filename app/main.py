import os
import sys
import json
import time
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Add project root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.common.config import ConfigManager
from src.common.logger import logger
from src.ingestion.markdown_loader import MarkdownLoader
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.txt_loader import TXTLoader
from src.chunking.hierarchical_chunker import MultiGranularityChunker
from src.embeddings.bge_embedding import SentenceTransformerEmbeddingProvider
from src.indexing.vector_store import VectorStoreManager
from src.generation.llama_cpp_provider import LlamaCppProvider
from src.retrieval.baseline_pipeline import BaselineRAGPipeline
from src.retrieval.qatm_pipeline import QATMRAGPipeline
from src.evaluation.evaluator import BenchmarkEvaluator

app = FastAPI(
    title="QATM-RAG Research API",
    description="Query-Adaptive Token-Minimizing RAG Framework API & Dashboard",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State Container
class SystemState:
    def __init__(self):
        self.cfg = ConfigManager()
        self.md_loader = MarkdownLoader()
        self.pdf_loader = PDFLoader()
        self.txt_loader = TXTLoader()
        self.chunker = MultiGranularityChunker()
        self.embedder = SentenceTransformerEmbeddingProvider(
            model_name=self.cfg.get("embedding.model_name", "BAAI/bge-small-en-v1.5"),
            device="cpu"
        )
        self.vector_store = VectorStoreManager(
            persist_directory="data/chroma_db",
            embedding_provider=self.embedder
        )
        self.llm = LlamaCppProvider(
            model_path=self.cfg.get("llm.model_path"),
            ollama_model=self.cfg.get("llm.ollama_model"),
            ollama_host=self.cfg.get("llm.ollama_host"),
            prefer_ollama=self.cfg.get("llm.prefer_ollama", False)
        )
        self.baseline_pipeline = BaselineRAGPipeline(self.vector_store, self.embedder, self.llm, top_k=5)
        self.qatm_pipeline = QATMRAGPipeline(self.vector_store, self.embedder, self.llm)
        self.active_doc_name: str = "ieee_project_dynamic_rag_architecture.md"
        self._auto_index_default()

    def _auto_index_default(self):
        default_doc = "ieee_project_dynamic_rag_architecture.md"
        if os.path.exists(default_doc):
            try:
                doc = self.md_loader.load(default_doc)
                chunks = self.chunker.chunk_document(doc)
                self.vector_store.clear()
                self.vector_store.add_chunks(chunks)
                self.active_doc_name = default_doc
                logger.info(f"Default document '{default_doc}' indexed ({len(chunks)} chunks).")
            except Exception as e:
                logger.warning(f"Failed to auto-index default document: {e}")

state: Optional[SystemState] = None

def get_state() -> SystemState:
    global state
    if state is None:
        state = SystemState()
    return state

# Request / Response Schemas
class QueryRequest(BaseModel):
    query: str
    document_path: Optional[str] = None

class BenchmarkRequest(BaseModel):
    dataset_path: Optional[str] = "data/evaluation/eval_dataset.json"

# Mount static folder
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def read_root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(
        content={"message": "QATM-RAG API running. UI dashboard loading..."},
        status_code=200
    )

@app.get("/api/health")
async def health_check():
    s = get_state()
    count = s.vector_store.get_count()
    return {
        "status": "online",
        "active_document": s.active_doc_name,
        "indexed_chunks": count,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "llm_provider": getattr(s.llm, "provider_name", "unknown"),
        "llm_is_real": getattr(s.llm, "is_real_llm", False)
    }

@app.post("/api/ingest")
async def ingest_document(file: UploadFile = File(None), file_path: Optional[str] = None):
    s = get_state()
    target_path = ""
    
    if file:
        save_dir = "data/uploads"
        os.makedirs(save_dir, exist_ok=True)
        target_path = os.path.join(save_dir, file.filename)
        with open(target_path, "wb") as f:
            content = await file.read()
            f.write(content)
    elif file_path and os.path.exists(file_path):
        target_path = file_path
    else:
        raise HTTPException(status_code=400, detail="Invalid file upload or file path provided.")

    ext = os.path.splitext(target_path)[1].lower()
    if ext == ".md":
        doc = s.md_loader.load(target_path)
    elif ext == ".pdf":
        doc = s.pdf_loader.load(target_path)
    elif ext == ".txt":
        doc = s.txt_loader.load(target_path)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file extension: {ext}")

    chunks = s.chunker.chunk_document(doc)
    s.vector_store.clear()
    s.vector_store.add_chunks(chunks)
    s.active_doc_name = os.path.basename(target_path)

    return {
        "message": "Document indexed successfully",
        "document_name": s.active_doc_name,
        "total_chunks": len(chunks),
        "total_documents_in_vector_store": s.vector_store.get_count()
    }

@app.post("/api/query")
async def run_query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    s = get_state()
    
    # 1. Run Baseline RAG Pipeline
    b_start = time.time()
    b_res = s.baseline_pipeline.run(req.query)
    b_latency = round((time.time() - b_start) * 1000, 2)

    # 2. Run QATM Dynamic Pipeline
    q_start = time.time()
    q_res = s.qatm_pipeline.run(req.query, baseline_tokens=b_res.context_tokens)
    q_latency = round((time.time() - q_start) * 1000, 2)

    # Calculate similarity score gradient steps for UI curve
    similarity_dropoffs = []
    deltas = q_res.trace_info.get("deltas", [])
    for i, d in enumerate(deltas):
        similarity_dropoffs.append({
            "rank_pair": f"{i+1} -> {i+2}",
            "delta": round(d, 4)
        })

    return {
        "query": req.query,
        "baseline": {
            "retrieved_chunks_count": b_res.retained_chunk_count,
            "context_tokens": b_res.context_tokens,
            "latency_ms": b_res.total_latency_ms,
            "answer": b_res.answer,
            "retrieved_chunks": [
                {
                    "id": s.get("chunk_id", f"chunk_{i}"),
                    "granularity": s.get("granularity", "standard"),
                    "text": s.get("document_name", ""),
                    "score": round(s.get("score", 0.0) or 0.0, 4)
                } for i, s in enumerate(b_res.sources)
            ]
        },
        "qatm": {
            "query_tier": q_res.query_tier.name,
            "query_tier_code": q_res.query_tier.value,
            "retrieved_chunks_count": q_res.retained_chunk_count,
            "context_tokens": q_res.context_tokens,
            "token_savings_percent": q_res.token_savings_percent,
            "latency_ms": q_res.total_latency_ms,
            "answer": q_res.answer,
            "similarity_dropoffs": similarity_dropoffs,
            "retrieved_chunks": [
                {
                    "id": s.get("chunk_id", f"chunk_{i}"),
                    "granularity": s.get("granularity", "standard"),
                    "text": s.get("document_name", ""),
                    "score": round(s.get("score", 0.0) or 0.0, 4)
                } for i, s in enumerate(q_res.sources)
            ]
        }
    }

@app.post("/api/benchmark")
async def run_benchmark(req: BenchmarkRequest):
    dataset_path = req.dataset_path or "data/evaluation/eval_dataset.json"
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail=f"Benchmark dataset path not found: {dataset_path}")

    s = get_state()
    evaluator = BenchmarkEvaluator(s.baseline_pipeline, s.qatm_pipeline)
    report = evaluator.evaluate_dataset(dataset_path)

    return report
