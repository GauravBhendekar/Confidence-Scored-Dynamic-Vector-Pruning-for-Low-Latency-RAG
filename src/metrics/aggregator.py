import numpy as np
from typing import List, Dict, Any
from src.common.types import PipelineResponse

class MetricsAggregator:
    @staticmethod
    def aggregate_responses(responses: List[PipelineResponse]) -> Dict[str, Any]:
        if not responses:
            return {}

        context_tokens = [r.context_tokens for r in responses]
        latencies = [r.total_latency_ms for r in responses]
        token_savings = [r.token_savings_percent or 0.0 for r in responses]
        retained_chunks = [r.retained_chunk_count for r in responses]

        return {
            "total_queries": len(responses),
            "context_tokens": {
                "mean": round(float(np.mean(context_tokens)), 2),
                "median": round(float(np.median(context_tokens)), 2),
                "p95": round(float(np.percentile(context_tokens, 95)), 2),
                "min": int(np.min(context_tokens)),
                "max": int(np.max(context_tokens))
            },
            "token_savings_percent": {
                "mean": round(float(np.mean(token_savings)), 2),
                "median": round(float(np.median(token_savings)), 2),
                "p95": round(float(np.percentile(token_savings, 95)), 2)
            },
            "latency_ms": {
                "mean": round(float(np.mean(latencies)), 2),
                "median": round(float(np.median(latencies)), 2),
                "p95": round(float(np.percentile(latencies, 95)), 2)
            },
            "avg_retained_chunks": round(float(np.mean(retained_chunks)), 2)
        }
