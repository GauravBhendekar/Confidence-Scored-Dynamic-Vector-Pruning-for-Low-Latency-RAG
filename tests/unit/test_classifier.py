import pytest
from src.common.types import QueryTier
from src.classification.classifier import QueryComplexityClassifier

def test_query_complexity_classification():
    classifier = QueryComplexityClassifier()

    # Tier 1 Factoid test
    res1 = classifier.classify("When was ChromaDB founded?")
    assert res1.predicted_tier == QueryTier.TIER_1_FACTOID

    # Tier 2 Descriptive test
    res2 = classifier.classify("Explain the process of vector normalization in RAG systems.")
    assert res2.predicted_tier == QueryTier.TIER_2_DESCRIPTIVE

    # Tier 3 Complex Synthesis test
    res3 = classifier.classify("Compare Naive RAG versus Dynamic Pruned RAG across latency and token savings.")
    assert res3.predicted_tier == QueryTier.TIER_3_COMPLEX
