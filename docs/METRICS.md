# QATM-RAG Research Metrics Specification

## Primary Research Metrics
1. **Token Savings Percentage (%):**
   $$\text{Token Savings \%} = \left( \frac{\text{Tokens}_{\text{Baseline}} - \text{Tokens}_{\text{QATM}}}{\text{Tokens}_{\text{Baseline}}} \right) \times 100$$
2. **Context Token Reduction:** Mean, Median, and P95 context token counts.
3. **Latency Breakdown:**
   - Retrieval Latency (ms)
   - Pruning Latency (ms)
   - Generation Latency (ms)
   - Total End-to-End Latency (ms)
4. **Retrieval Depth:** Average retained chunk count (QATM) vs fixed `k` (baseline).

## Validity Gating
Every benchmark report includes `summary.llm_status`:
- `backend` — the backend actually used (`llama_cpp (...)`, `ollama (...)`, or
  `extractive_fallback (no LLM)`).
- `is_real_llm` — whether a real model generated the answers.
- `metrics_valid` — `false` when the extractive fallback ran. Latency,
  token-savings, and quality figures are **not** publishable in that case.
- `warmed_up` — whether `warm_up()` ran before measurement. Cold-start latency
  (observed ~62 s for a 3B model) otherwise dominates the mean and P95 and must
  be excluded.

## Answer Quality Metrics — **NOT YET IMPLEMENTED**
The framework currently measures **cost** metrics only (tokens, latency, depth).
It does **not** yet measure whether answer quality is preserved, which the paper
requires in order to claim token savings come "without degrading accuracy."

Planned:
- Ragas `Faithfulness`, `Context Precision`, `Context Recall`
  (Ragas is **not** in `requirements.txt` yet).
- Reference-based `Exact Match` / `F1` against dataset gold answers.
- Hallucination rate.

Until at least one of these exists, results should be reported as cost
reductions only, not as a quality-preserving result.

