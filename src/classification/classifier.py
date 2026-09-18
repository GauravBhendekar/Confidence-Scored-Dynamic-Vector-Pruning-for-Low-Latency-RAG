import re
from src.common.types import QueryTier, ClassificationResult
from src.common.logger import logger

class QueryComplexityClassifier:
    def __init__(self, confidence_threshold: float = 0.70):
        self.confidence_threshold = confidence_threshold

        # Tier 1 Rules (Factoid / Entity / Year / Definition)
        self.factoid_patterns = [
            r"^\s*(who|when|where|what is the name|what year|what date|define|which)\b",
            r"\b(definition|founded|invented|created|born|located|capital|author|price|cost)\b"
        ]

        # Tier 3 Rules (Comparison / Multi-hop Synthesis / Analytical Trade-offs)
        self.complex_patterns = [
            r"\b(compare|versus|vs|relationship|differences|pros and cons|advantages and disadvantages|trade-offs|synthesize|analyze|impact of .* on)\b",
            r"\b(how do .* and .* differ|all factors|multi-step|evaluate)\b"
        ]

    def classify(self, query: str) -> ClassificationResult:
        query_clean = query.strip().lower()

        # Check Tier 3 first
        complex_matches = sum(1 for pattern in self.complex_patterns if re.search(pattern, query_clean))
        if complex_matches > 0 or (" and " in query_clean and len(query_clean.split()) > 12):
            confidence = min(0.75 + 0.1 * complex_matches, 0.95)
            return ClassificationResult(
                query=query,
                predicted_tier=QueryTier.TIER_3_COMPLEX,
                confidence=confidence,
                reasoning=f"Matched complex synthesis patterns ({complex_matches} indicators).",
                is_fallback=False
            )

        # Check Tier 1
        factoid_matches = sum(1 for pattern in self.factoid_patterns if re.search(pattern, query_clean))
        if factoid_matches > 0 and len(query_clean.split()) <= 10:
            confidence = min(0.70 + 0.1 * factoid_matches, 0.92)
            return ClassificationResult(
                query=query,
                predicted_tier=QueryTier.TIER_1_FACTOID,
                confidence=confidence,
                reasoning=f"Matched factoid/entity patterns ({factoid_matches} indicators).",
                is_fallback=False
            )

        # Tier 2 Default (Descriptive / Explanatory)
        if query_clean.startswith(("how", "why", "explain", "describe")):
            return ClassificationResult(
                query=query,
                predicted_tier=QueryTier.TIER_2_DESCRIPTIVE,
                confidence=0.88,
                reasoning="Query asks for process/explanation ('how'/'why'/'explain').",
                is_fallback=False
            )

        # Low confidence fallback scenario
        logger.warning(f"Classification confidence low for query '{query}'. Falling back to Tier 2 (Descriptive).")
        return ClassificationResult(
            query=query,
            predicted_tier=QueryTier.TIER_2_DESCRIPTIVE,
            confidence=0.55,
            reasoning="Uncertain query intent. Safe fallback to Tier 2 Descriptive.",
            is_fallback=True
        )
