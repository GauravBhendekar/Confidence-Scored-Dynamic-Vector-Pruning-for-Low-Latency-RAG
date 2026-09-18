# QATM-RAG Architecture Documentation

## System Overview
Query-Adaptive Token-Minimizing RAG (QATM-RAG) is a low-latency, research-grade
RAG framework designed to minimize LLM context token usage without degrading
answer accuracy.

## Component Breakdown
1. **Multi-Granularity Document Indexer:** Splits documents into 3 linked
   structural levels:
   - Level 3: Parent Blocks (600–800 tokens)
   - Level 2: Standard Chunks (150–250 tokens)
   - Level 1: Micro-Spans (15–40 tokens / Sentence)
2. **Query Complexity Classifier:** Categorizes incoming queries into 3 Tiers
   (Factoid, Descriptive, Complex) with confidence scoring and fallback rules.
3. **Dynamic Score Gradient Pruner:** Calculates cosine similarity drop-offs
   ($\Delta_i = s_i - s_{i+1}$) to locate knee-points and cut off noisy tail
   vectors.
4. **Sentence Context Trimmer:** Extracts exact sentence micro-spans matching
   query terms for Tier 1 factoid queries.
5. **Token Budget Controller:** Enforces hard context token caps per tier
   (Tier 1: 200 max, Tier 2: 700 max, Tier 3: 1200 max).
6. **Minimal Prompt Assembler:** Combines pruned context blocks with zero-shot
   instruction templates.
7. **LLM Provider Abstraction:** Resolves a generation backend in the order
   **llama-cpp GGUF → Ollama → extractive fallback**, and reports honestly which
   one is active. Target local model: Llama-3.2-3B (Q4_K_M GGUF) or
   `llama3.2` via Ollama.

## Retrieval Granularity Policy

| Tier | Granularity | Candidate pool | Retained `k` | Token cap |
| :--- | :--- | ---: | ---: | ---: |
| Tier 1 Factoid | `MICRO_SPAN` | 15 | 1–2 | 200 |
| Tier 2 Descriptive | `STANDARD` | 15 | 2–4 | 700 |
| Tier 3 Complex | `STANDARD` | 15 | 3–5 | 1200 |

**Why Tier 3 uses standard chunks, not parent blocks.** An earlier revision
retrieved parent blocks (~700 tokens each) for Tier 3 while the baseline
retrieved standard chunks (~200 tokens each). That made "complex" queries
consume far *more* context than the baseline, inverting the token-minimization
objective and producing negative savings (−162% to −197% on the benchmark).
Tier 3 now retrieves the same granularity as the baseline; additional depth for
complex queries comes from a wider candidate pool, and `max_k` is pinned to the
baseline's `k=5` so QATM can never retain more chunks than it is compared against.

## Candidate Pool Hygiene
Because every paragraph is indexed at all three granularities, the same sentence
can appear in a candidate pool more than once (once as a standard chunk, once as
a micro-span). `QATMRAGPipeline._dedupe_by_text()` drops normalized-text
duplicates after retrieval so the token budget is spent on *distinct* evidence
rather than repeated content.

## Pruning Decision Procedure
1. Sort candidates by descending cosine similarity.
2. Compute $\Delta_i = s_i - s_{i+1}$.
3. **Absolute score gate:** cut at the first chunk scoring below `0.50`.
4. **Information cliff:** cut at the first $\Delta \geq 0.15$.
5. **Flat-distribution widening (Tier 3 only):** if the maximum $\Delta$ is below
   the cliff threshold, the evidence is spread evenly rather than concentrated
   behind one knee. For complex queries this is the multi-hop case, so the
   pruner widens to `max_k` instead of clamping down. Reported as
   `strategy_used="flat_widened"`.
6. **Tier constraint:** clamp to `min_k <= k <= max_k`.

## LLM Backend Resolution
1. **llama-cpp-python** — used if a GGUF file exists at `llm.model_path`.
2. **Ollama** — used if `GET /api/tags` is reachable and serves the configured
   `llm.ollama_model` (matching ignores the `:latest` tag). Endpoint and model
   are configurable via `llm.ollama_host` / `llm.ollama_model`.
3. **Extractive fallback** — deterministic keyword extraction. No model runs.
   Logged as a loud warning, and `is_real_llm = False`.

`LlamaCppProvider.provider_name` and `.is_real_llm` expose the active backend.
`warm_up()` performs one throwaway generation to load the model into memory
before any benchmarking, so the first measured query does not absorb cold-start
latency (which was observed to be ~62 s for a 3B model, enough to corrupt the
mean and P95).

## Hardware Optimization Strategy
- **GPU VRAM:** GGUF Q4 model fits in ~2.1 GB VRAM (RTX 3050 Laptop GPU 4 GB
  VRAM compatible).
- **RAM:** Sub-100 MB embedding memory footprint (`BAAI/bge-small-en-v1.5`,
  384-dim).
