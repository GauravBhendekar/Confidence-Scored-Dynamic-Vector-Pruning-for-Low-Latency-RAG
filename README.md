# Query-Adaptive Token-Minimizing RAG (QATM-RAG)

An advanced, low-latency, research-grade Retrieval-Augmented Generation (RAG) framework designed to eliminate LLM context token bloat and prompt pollution by dynamically classifying query complexity, scoring vector similarity gradients ($\Delta$), and pruning tail noise.

---

## 🏗️ Architecture & Features

- **Multi-Granularity Document Indexer:** Level 1 Micro-Spans (Sentences), Level 2 Standard Chunks (Paragraphs), and Level 3 Parent Blocks (Sections) with bidirectional lineage metadata.
- **Query Complexity Classifier:** 3 Tiers (Factoid, Descriptive, Complex Synthesis) with confidence scoring and fallback rules.
- **Dynamic Score Gradient Pruner:** Calculates $\Delta_i = s_i - s_{i+1}$ to identify knee-points and remove low-confidence tail chunks. Detects *flat* score curves on complex queries and widens retrieval instead of truncating multi-hop evidence.
- **Candidate Pool Dedup:** Removes near-duplicate text returned at multiple granularities so the token budget covers distinct evidence.
- **Sentence Context Trimmer:** Extracts precise sentence spans for factoid queries (Tier 1).
- **Token Budget Controller:** Enforces hard token caps per query tier (Tier 1: 200, Tier 2: 700, Tier 3: 1200).
- **LLM Provider Abstraction:** Resolves GGUF → Ollama → extractive fallback, reports which backend is active, and warms the model up before benchmarking.
- **Side-by-Side Research Evaluator:** Compares QATM-RAG against standard static Top-K=5 Baseline RAG, with a `llm_status` validity gate in the report.

---

## 🚀 Quick Start & Reproduction

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Set Up a Local LLM (required for valid benchmarks)
```bash
ollama pull llama3.2
```
Ollama must be running on `http://localhost:11434`. Configure via
`llm.ollama_host` / `llm.ollama_model` in `configs/default_config.yaml`, or the
`OLLAMA_HOST` / `OLLAMA_MODEL` environment variables.

> If no LLM is available the framework silently falls back to a deterministic
> extractive stub. The benchmark still runs, but `summary.llm_status.metrics_valid`
> will be `false` and the latency/quality numbers are **not** valid.

### 3. Run Unit Tests
```bash
python -m pytest tests/unit
```

### 4. Run Baseline RAG Pipeline
```bash
python run_baseline.py --dataset data/evaluation/eval_dataset.json --doc ieee_project_dynamic_rag_architecture.md
```

### 5. Run QATM-RAG Experimental Pipeline
```bash
python run_qatm.py --dataset data/evaluation/eval_dataset.json --doc ieee_project_dynamic_rag_architecture.md
```

### 6. Run Full Side-by-Side Research Benchmark
```bash
python run_benchmark.py --dataset data/evaluation/eval_dataset.json --doc ieee_project_dynamic_rag_architecture.md
```

### 7. Launch the Web Dashboard
```bash
uvicorn app.main:app --reload
```
Then open `http://localhost:8000`. `GET /api/health` reports the active
`llm_provider` and whether it is a real model.

### Measured Results (llama3.2, 6 queries)

| Metric | Baseline (k=5) | QATM-RAG | Change |
| :--- | ---: | ---: | ---: |
| Mean context tokens | 289.83 | 232.0 | −20.5% |
| Mean total latency | 4,064 ms | 3,744 ms | −7.9% |
| Avg retained chunks | 5.0 | 4.0 | −1 |

**Caveats:** only 6 queries on a single source document; the indexed document is
this project's own proposal, which contains the paper's *expected* result table.
A neutral corpus (HotpotQA / SQuAD / MS-MARCO) is still needed before these
numbers support a publication claim. See `PROJECT_STATUS.md`.


---

## 📊 Project Directory Structure

```
.
├── app/                  # Web API & UI Dashboard
├── configs/              # YAML Experiment Configuration files
├── data/                 # Raw, processed, and evaluation datasets
├── docs/                 # Architectural, Data Flow, and Metrics Documentation
├── src/                  # Core Modular Source Code
│   ├── chunking/         # Hierarchical multi-granularity chunker
│   ├── classification/   # Query complexity classifier
│   ├── common/           # Data types, logger, config manager
│   ├── embeddings/       # SentenceTransformer provider
│   ├── evaluation/       # Benchmark evaluator
│   ├── generation/       # LLM provider wrapper
│   ├── indexing/         # Vector store manager (ChromaDB)
│   ├── metrics/          # Metrics aggregator
│   ├── prompting/        # Minimal prompt assembler & token budget
│   ├── pruning/          # Dynamic score gradient pruner
│   ├── retrieval/        # Baseline and QATM pipeline engines
│   └── trimming/         # Micro-span sentence trimmer
├── tests/                # Unit and Integration test suites
├── run_baseline.py       # Control experiment script
├── run_qatm.py           # Active QATM experiment script
├── run_benchmark.py      # Side-by-side evaluator script
└── requirements.txt      # Python dependencies
```
