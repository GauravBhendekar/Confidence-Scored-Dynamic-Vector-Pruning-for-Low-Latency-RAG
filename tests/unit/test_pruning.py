import pytest
from src.common.types import Chunk, Granularity, QueryTier
from src.pruning.gradient_pruner import DynamicScoreGradientPruner

def test_gradient_knee_pruning():
    pruner = DynamicScoreGradientPruner(gradient_threshold=0.15, absolute_min_score=0.50)

    # Simulated retrieved candidate pool (high similarity then sharp drop)
    chunks = [
        Chunk(chunk_id="1", document_id="d1", document_name="d.txt", granularity=Granularity.STANDARD, text="c1", token_count=10, ordering_index=0, similarity_score=0.92),
        Chunk(chunk_id="2", document_id="d1", document_name="d.txt", granularity=Granularity.STANDARD, text="c2", token_count=10, ordering_index=1, similarity_score=0.90),
        Chunk(chunk_id="3", document_id="d1", document_name="d.txt", granularity=Granularity.STANDARD, text="c3", token_count=10, ordering_index=2, similarity_score=0.88),
        Chunk(chunk_id="4", document_id="d1", document_name="d.txt", granularity=Granularity.STANDARD, text="c4", token_count=10, ordering_index=3, similarity_score=0.60), # Sharp drop: delta = 0.28
        Chunk(chunk_id="5", document_id="d1", document_name="d.txt", granularity=Granularity.STANDARD, text="c5", token_count=10, ordering_index=4, similarity_score=0.58),
    ]

    # Run pruning for Tier 2 Descriptive (min=2, max=4)
    res = pruner.prune(chunks, QueryTier.TIER_2_DESCRIPTIVE)

    assert res.retained_count == 3 # Should truncate at index 3 before the 0.28 drop
    assert res.knee_point_index == 2 # Delta max is between item 3 (0.88) and item 4 (0.60)
    assert len(res.retained_chunks) == 3
    assert res.retained_chunks[-1].similarity_score == 0.88


def test_tier3_cap_keeps_token_savings_possible():
    """Tier 3 must cap retained chunks well below the fixed-k=5 baseline pool so
    it never retrieves more chunks than the baseline it is compared against."""
    pruner = DynamicScoreGradientPruner()

    # Flat, high-confidence pool: no cliff, no absolute gate trip, so the tier cap decides.
    chunks = [
        Chunk(
            chunk_id=str(i),
            document_id="d1",
            document_name="d.txt",
            granularity=Granularity.STANDARD,
            text=f"chunk {i}",
            token_count=100,
            ordering_index=i,
            similarity_score=0.90 - (0.005 * i),
        )
        for i in range(15)
    ]

    res = pruner.prune(chunks, QueryTier.TIER_3_COMPLEX)

    assert res.retained_count <= 5, "Tier 3 must not exceed the k=5 baseline chunk count"
    assert res.retained_count == 5


def test_tier3_flat_distribution_widens():
    """A flat score curve on a complex query means spread-out evidence; Tier 3
    should widen to its max rather than clamp down and drop needed facts."""
    pruner = DynamicScoreGradientPruner(gradient_threshold=0.15)

    # Flat curve: max delta ~0.01, far below the cliff threshold.
    chunks = [
        Chunk(
            chunk_id=str(i),
            document_id="d1",
            document_name="d.txt",
            granularity=Granularity.STANDARD,
            text=f"chunk {i}",
            token_count=50,
            ordering_index=i,
            similarity_score=0.75 - (0.01 * i),
        )
        for i in range(15)
    ]

    res = pruner.prune(chunks, QueryTier.TIER_3_COMPLEX)

    assert res.strategy_used == "flat_widened"
    assert res.retained_count == 5


def test_cliff_still_truncates_for_tier3():
    """When a genuine cliff exists, Tier 3 must still cut at the knee."""
    pruner = DynamicScoreGradientPruner(gradient_threshold=0.15)

    chunks = [
        Chunk(chunk_id=str(i), document_id="d1", document_name="d.txt",
              granularity=Granularity.STANDARD, text=f"c{i}", token_count=50,
              ordering_index=i, similarity_score=s)
        for i, s in enumerate([0.90, 0.88, 0.86, 0.60, 0.58, 0.57, 0.56])
    ]

    res = pruner.prune(chunks, QueryTier.TIER_3_COMPLEX)

    assert res.strategy_used == "gradient_knee_tier_constrained"
    assert res.retained_count == 3
