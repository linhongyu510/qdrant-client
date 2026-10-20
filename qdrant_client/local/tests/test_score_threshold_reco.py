import pytest

from qdrant_client import QdrantClient, models


def _make(distance):
    client = QdrantClient(":memory:")
    client.create_collection(
        "c", vectors_config=models.VectorParams(size=2, distance=distance)
    )
    client.upsert(
        "c",
        points=[
            models.PointStruct(id=1, vector=[0.0, 0.0]),
            models.PointStruct(id=2, vector=[1.0, 0.0]),
            models.PointStruct(id=3, vector=[2.0, 0.0]),
            models.PointStruct(id=4, vector=[3.0, 0.0]),
            models.PointStruct(id=5, vector=[10.0, 0.0]),
        ],
    )
    return client


@pytest.mark.parametrize("distance", [models.Distance.EUCLID, models.Distance.MANHATTAN])
def test_best_score_threshold_not_inverted_on_smaller_is_better(distance):
    """https://github.com/qdrant/qdrant-client/issues/1370

    Recommend best_score is a bigger-is-better sigmoid even on a Euclidean/Manhattan collection,
    so score_threshold must keep points above the threshold instead of dropping the whole set on
    the first (best-scoring) point.
    """
    client = _make(distance)
    query = models.RecommendQuery(
        recommend=models.RecommendInput(positive=[1], strategy=models.RecommendStrategy.BEST_SCORE)
    )

    pts = client.query_points("c", query=query, limit=10, score_threshold=0.1).points
    ids = [p.id for p in pts]

    # Closest point (id=2) has the highest sigmoid score (~0.25) and must survive.
    assert 2 in ids, f"score_threshold dropped the best-scoring point: {ids}"
    # Farthest point (id=5) has the lowest sigmoid score (~0.005) and must be excluded.
    assert 5 not in ids, f"score_threshold kept a sub-threshold point: {ids}"


@pytest.mark.parametrize("distance", [models.Distance.EUCLID, models.Distance.MANHATTAN])
def test_sum_scores_threshold_is_bigger_is_better(distance):
    client = _make(distance)
    query = models.RecommendQuery(
        recommend=models.RecommendInput(positive=[1], strategy=models.RecommendStrategy.SUM_SCORES)
    )
    # sum_scores yields negative distances (closer = larger). Threshold -5 keeps id=2 (-1) and
    # id=3 (-4/-2) but drops id=4 and id=5.
    pts = client.query_points("c", query=query, limit=10, score_threshold=-5).points
    ids = [p.id for p in pts]
    assert 2 in ids and 3 in ids, ids
    assert 5 not in ids, ids
