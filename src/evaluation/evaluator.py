import json
import os
from typing import List, Dict, Any
from src.retrieval.baseline_pipeline import BaselineRAGPipeline
from src.retrieval.qatm_pipeline import QATMRAGPipeline
from src.metrics.aggregator import MetricsAggregator
from src.common.logger import logger

class BenchmarkEvaluator:
    def __init__(self, baseline_pipeline: BaselineRAGPipeline, qatm_pipeline: QATMRAGPipeline):
        self.baseline = baseline_pipeline
        self.qatm = qatm_pipeline

    def _llm_status(self) -> Dict[str, Any]:
        """Report which generation backend was active so results are not
        mistaken for real-LLM measurements when the extractive fallback ran."""
        llm = getattr(self.baseline, "llm_provider", None)
        is_real = bool(getattr(llm, "is_real_llm", False))
        name = getattr(llm, "provider_name", "unknown")
        if not is_real:
            logger.warning(
                "Benchmark is running with the extractive fallback (no real LLM). "
                "Latency and answer-quality metrics are NOT valid for research claims."
            )
        return {
            "backend": name,
            "is_real_llm": is_real,
            "metrics_valid": is_real
        }

    def evaluate_dataset(self, dataset_path: str, output_dir: str = "results") -> Dict[str, Any]:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

        with open(dataset_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        # Warm up the LLM so the first measured query does not absorb model
        # cold-start latency (which would otherwise skew mean/P95 by tens of seconds).
        llm = getattr(self.baseline, "llm_provider", None)
        warmed = False
        if llm is not None and hasattr(llm, "warm_up"):
            warmed = llm.warm_up()

        logger.info(f"Starting evaluation run on {len(items)} queries from '{dataset_path}'...")

        baseline_responses = []
        qatm_responses = []
        detailed_records = []

        for idx, item in enumerate(items):
            query = item["question"]
            expected_tier = item.get("expected_tier", 2)

            # Run Baseline RAG
            base_res = self.baseline.run(query)
            baseline_responses.append(base_res)

            # Run QATM-RAG with baseline token baseline context
            qatm_res = self.qatm.run(query, baseline_tokens=base_res.context_tokens)
            qatm_responses.append(qatm_res)

            record = {
                "id": item.get("id", f"q_{idx}"),
                "question": query,
                "expected_tier": expected_tier,
                "baseline": {
                    "context_tokens": base_res.context_tokens,
                    "latency_ms": base_res.total_latency_ms,
                    "retrieved_chunks": base_res.retrieved_chunk_count,
                    "answer": base_res.answer
                },
                "qatm": {
                    "detected_tier": qatm_res.query_tier.value,
                    "context_tokens": qatm_res.context_tokens,
                    "token_savings_pct": qatm_res.token_savings_percent,
                    "latency_ms": qatm_res.total_latency_ms,
                    "retained_chunks": qatm_res.retained_chunk_count,
                    "answer": qatm_res.answer
                }
            }
            detailed_records.append(record)

        # Aggregate Metrics
        base_stats = MetricsAggregator.aggregate_responses(baseline_responses)
        qatm_stats = MetricsAggregator.aggregate_responses(qatm_responses)

        report = {
            "summary": {
                "total_queries_evaluated": len(items),
                "llm_status": {**self._llm_status(), "warmed_up": warmed},
                "baseline_stats": base_stats,
                "qatm_stats": qatm_stats
            },
            "detailed_results": detailed_records
        }

        os.makedirs(output_dir, exist_ok=True)
        report_file = os.path.join(output_dir, "benchmark_report.json")
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Evaluation completed. Benchmark report saved to '{report_file}'.")
        return report
