# 📊 Project Progress & Status Report: QATM-RAG

**Project Name:** Query-Adaptive Token-Minimizing RAG (QATM-RAG)  
**Target Publication:** IEEE Access / IEEE CAI / IEEE Big Data  
**Current Overall Completion:** **`88%`**  
**Last Updated:** 2026-09-18

---

## 📈 Status Overview

```
Core Algorithmic Engine:     [████████████████████] 98%   (Tier 3 retrieval fixed)
Web API & UI Dashboard:      [████████████████████] 100%  ✅
Experiment Scripts:          [████████████████████] 100%  ✅
Documentation & Specs:       [██████████████████░░] 90%
Unit & Integration Tests:    [████████████░░░░░░░░] 62%   (9 unit tests passing)
Benchmark Datasets:          [██████░░░░░░░░░░░░░░] 30%   ⚠️  BLOCKING
Real-LLM Benchmarking:       [████████████████████] 100%  ✅  (Ollama llama3.2)
```

---

## 🔧 Recent Fixes (2026-09-18)

Three correctness issues were found and fixed. All were verified with a real
LLM (`llama3.2` via Ollama), not the extractive fallback.

### Fix 1 — Tier 3 token inflation (critical correctness bug)

**Symptom.** Complex (Tier 3) queries showed **negative** token savings
(−162% and −197% in the benchmark) because the pipeline retrieved *parent
blocks* (~700 tokens each) while the baseline retrieved *standard chunks*
(~200 tokens each). "Complex" queries therefore consumed far more context
than the baseline they were compared against, inverting the paper's core claim.

**Fix.**
- `src/retrieval/qatm_pipeline.py` — Tier 3 now retrieves `STANDARD`
  granularity, not `PARENT_BLOCK`. Depth comes from a wider candidate pool,
  not larger chunks.
- `src/pruning/gradient_pruner.py` — Tier 3 `max_k` lowered `6 → 5` so QATM
  can never retain more chunks than the fixed-`k=5` baseline.
- `src/prompting/token_budget.py` — Tier 3 cap lowered `1800 → 1200` tokens.
- `configs/qatm.yaml` — Tier 3 `max_tokens` 1800→1200, `max_items` 6→5,
  `retrieval_granularity` parent_block→standard.

**Result.** Both Tier 3 queries moved from −162%/−197% to **0%** (retaining
5 chunks, same as baseline). Mean savings went **−39.5% → +20.5%**.

### Fix 2 — Real LLM integration (benchmark validity)

**Symptom.** No LLM was ever running. `LlamaCppProvider` silently fell back to
a keyword-extraction stub, producing a bogus ~4,150 ms latency for both
pipelines and no possibility of measuring answer quality.

**Fix.** `src/generation/llama_cpp_provider.py` rewritten with a resolution
order of **GGUF → Ollama → extractive fallback**:
- Startup probe of `GET /api/tags` that verifies the target model is actually
  pulled (matches ignoring the `:latest` tag suffix).
- Real generation options (`num_predict`, `temperature`) and resolved model name.
- New `provider_name` and `is_real_llm` properties; `health_check()` now returns
  `is_real_llm` instead of a misleading "model loaded" flag.
- Loud warning when running in fallback mode.
- `warm_up()` method that runs one throwaway generation to load the model into
  memory before measurement.

Config wired through all call sites (`run_baseline.py`, `run_qatm.py`,
`run_benchmark.py`, `app/main.py`); `configs/default_config.yaml` gained
`ollama_host`, `ollama_model`, `prefer_ollama`.

**Result.** Benchmark logs `Using Ollama backend ... model 'llama3.2:latest'`.
Baseline and QATM latencies are now genuinely different and measurable.

### Fix 3 — Cold-start skew + duplicate chunks + flat-distribution under-retrieval

**Cold-start.** The first baseline query absorbed ~62 s of model load time,
inflating the baseline mean from ~4,900 ms to 14,012 ms. `BenchmarkEvaluator`
now calls `llm.warm_up()` before the measured loop and records `warmed_up` in
the report. Baseline mean corrected to **4,064 ms**.

**Duplicate chunks.** Every paragraph is indexed at all three granularities, so
the same sentence could occupy two slots in the candidate pool. Added
`QATMRAGPipeline._dedupe_by_text()` to drop normalized-text duplicates after
retrieval, so the retained budget covers *distinct* evidence.

**Flat-distribution under-retrieval (Tier 3).** When a complex query's score
curve is flat (max Δ well below the 0.15 cliff threshold), evidence is spread
evenly rather than concentrated behind one knee. The old logic silently clamped
to `min_k`. The pruner now detects `is_flat` for Tier 3 and widens to `max_k`,
recording `strategy_used="flat_widened"`. Genuine cliffs still truncate
(covered by a regression test).

---

## 📊 Verified Benchmark Results (real LLM, `llama3.2:latest`)

Source: `results/benchmark_report.json`, 6 queries,
`llm_status.metrics_valid = true`, `warmed_up = true`.

| Metric | Baseline (k=5) | QATM-RAG | Change |
| :--- | ---: | ---: | ---: |
| Mean context tokens | 289.83 | **232.0** | **−20.5%** |
| Med. context tokens | 292.5 | 218.0 | −25.4% |
| P95 context tokens | 352.25 | 352.25 | 0% |
| Mean total latency | 4,064 ms | **3,744 ms** | **−7.9%** |
| P95 total latency | 5,276 ms | 5,387 ms | +2.1% |
| Avg retained chunks | 5.0 | 4.0 | −1 |

Per-query token savings: q1 +2.0%, q2 **+71.5%**, q3 +23.0%, q4 +26.3%,
q5 0%, q6 0%.

### ⚠️ Honest caveats before citing these numbers

1. **Mean latency gap is modest (−7.9%) and P95 is slightly worse.** With only
   6 queries and one source document, the latency signal is weak. Token savings
   are consistent; latency needs a larger, longer-context corpus to show the
   prefill reduction the architecture predicts.
2. **q2 and q6 return "Information not found in context."** q2 asks about a
   framework absent from the indexed document (expected, and correct refusal
   behavior). q6 fails for **both** pipelines — the source document does not
   actually explain the relationship q6 asks about, so this is a
   dataset/document limit, not a pruning bug. It is a *retrieval-recall*
   finding worth reporting, not hiding.
3. **The benchmark indexes `ieee_project_dynamic_rag_architecture.md`, which
   contains the paper's own *expected* results table.** Generated answers can
   therefore quote those expected figures as if they were findings (this happens
   in q5). Use a neutral corpus for final measurements.
4. **6 queries is not an evaluation set.** See Phase 3 below.

---

## ✅ Completed Components & Modules

### 1. Core Source Code (`src/`) — **~98% Complete**
- **Ingestion Engine (`src/ingestion/`)**: `.pdf`, `.md`, `.txt` with cleaning &
  normalization.
- **Hierarchical Multi-Granularity Chunker (`src/chunking/`)**: 3 linked levels
  (Micro-Spans 15–40 tok, Standard 150–250 tok, Parent Blocks 600–800 tok) with
  bidirectional lineage metadata.
- **Embedding Provider (`src/embeddings/`)**: `BAAI/bge-small-en-v1.5` behind an
  abstract base class.
- **Vector Store Manager (`src/indexing/`)**: ChromaDB persistent store, cosine
  space, multi-level collections.
- **Query Complexity Classifier (`src/classification/`)**: 3 tiers with
  confidence scoring & fallback.
- **Dynamic Score Gradient Pruner (`src/pruning/`)**: Δ gradient knee detection,
  absolute score gate, cliff cutoff, tier-constrained `k`, and flat-distribution
  widening for Tier 3.
- **Sentence Context Trimmer (`src/trimming/`)**: micro-span extraction for Tier 1.
- **Prompt Builder & Token Controller (`src/prompting/`)**: hard caps per tier
  (Tier 1: 200, Tier 2: 700, Tier 3: 1200 tokens).
- **LLM Generation Abstraction (`src/generation/`)**: GGUF / Ollama / extractive
  fallback with honest backend reporting and warm-up.
- **Retrieval Engine (`src/retrieval/`)**: `baseline_pipeline.py` (static k=5)
  and `qatm_pipeline.py` (dynamic pruning + dedup).
- **Metrics Aggregator (`src/metrics/`, `src/evaluation/`)**: token savings %,
  latency mean/median/P95, retained chunks, LLM-status reporting.

### 2. Experiment Execution Scripts — **100% Complete**
- ✅ [run_baseline.py](file:///d:/3%20rd%20year/EDI/run_baseline.py)
- ✅ [run_qatm.py](file:///d:/3%20rd%20year/EDI/run_qatm.py)
- ✅ [run_benchmark.py](file:///d:/3%20rd%20year/EDI/run_benchmark.py)

### 3. Web API & UI Dashboard (`app/`) — **100% Complete ✅**
- ✅ `app/main.py` — FastAPI server: `POST /api/query`, `POST /api/benchmark`,
  `POST /api/ingest`, `GET /api/health`.
- ✅ `app/static/` — HTML/JS/CSS dashboard (token reduction, Δ drop-off curves,
  tier classification, side-by-side answers).
- Note: `/api/health` now reports `llm_provider` and `llm_is_real`.

### 4. Documentation — **90% Complete**
- ✅ [README.md](file:///d:/3%20rd%20year/EDI/README.md)
- ✅ [docs/ARCHITECTURE.md](file:///d:/3%20rd%20year/EDI/docs/ARCHITECTURE.md)
- ✅ [docs/DATA_FLOW.md](file:///d:/3%20rd%20year/EDI/docs/DATA_FLOW.md)
- ✅ [docs/METRICS.md](file:///d:/3%20rd%20year/EDI/docs/METRICS.md)
- ✅ [ieee_project_dynamic_rag_architecture.md](file:///d:/3%20rd%20year/EDI/ieee_project_dynamic_rag_architecture.md)

### 5. Unit Testing — **9/9 passing**
- ✅ `tests/unit/test_chunking.py` — hierarchy + lineage
- ✅ `tests/unit/test_classifier.py` — tier routing
- ✅ `tests/unit/test_pruning.py` — knee detection, Tier 3 cap, flat-widening,
  cliff truncation
- ✅ `tests/unit/test_app.py` — API health/query/benchmark

---

## 🚀 Next Steps & Roadmap (Remaining ~12%)

### Phase 3: Large-Scale IEEE Benchmark Datasets — **PRIORITY, currently blocking**
*Currently **30%** complete. This is the single biggest gap: 6 queries on one
document cannot support a publication claim.*

- [ ] **Task 3.1: Expand Evaluation Datasets (`data/evaluation/`)**
  - *HotpotQA* (multi-hop complex — exercises Tier 3)
  - *SQuAD 2.0* (factoid — exercises Tier 1)
  - *MS-MARCO* (descriptive — exercises Tier 2)
  - Use a **neutral corpus** (not this project's own proposal doc).
- [ ] **Task 3.2: Answer-quality metrics.** No faithfulness/accuracy scoring
  exists yet. Add Ragas (not currently in `requirements.txt`) or an equivalent
  EM/F1 score against reference answers, so the paper can claim quality is
  *preserved*, not just that tokens dropped.
- [ ] **Task 3.3: Automated Plotting (`scripts/generate_plots.py`)** — latency
  reduction bar charts and token-savings curves for the paper.
- [ ] **Task 3.4: Warm-up note in methodology.** Document that `warm_up()` runs
  before measurement, since cold-start would otherwise dominate latency.

### Phase 2: Remaining Tests
- [ ] `tests/unit/test_indexing.py`, `test_embeddings.py`, `test_ingestion.py`,
  `test_trimming.py`, `test_prompting.py`, `test_retrieval.py`
- [ ] `tests/integration/` — end-to-end ingestion → generation

---

## 🛠️ Verification & Execution Commands

```bash
# Unit tests
python -m pytest tests/unit

# One-time: pull the local model
ollama pull llama3.2

# Comparative benchmark (writes results/benchmark_report.json)
python run_benchmark.py --dataset data/evaluation/eval_dataset.json --doc ieee_project_dynamic_rag_architecture.md

# Individual pipelines
python run_baseline.py --dataset data/evaluation/eval_dataset.json --doc ieee_project_dynamic_rag_architecture.md
python run_qatm.py --dataset data/evaluation/eval_dataset.json --doc ieee_project_dynamic_rag_architecture.md

# Web API + dashboard
uvicorn app.main:app --reload
```

> **Note:** If Ollama is not running, the benchmark still completes but falls
> back to the extractive stub. Check `summary.llm_status.metrics_valid` in the
> report — if it is `false`, the latency and quality numbers are not valid.
