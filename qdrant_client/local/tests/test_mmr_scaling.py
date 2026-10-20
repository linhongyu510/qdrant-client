from contextlib import closing
from typing import Any

import numpy as np
import pytest

from qdrant_client import AsyncQdrantClient, QdrantClient, models
from qdrant_client.local import local_collection


@pytest.mark.parametrize(
    ("kind", "distance"),
    [
        (kind, distance)
        for kind in ("dense", "multi")
        for distance in (models.Distance.DOT, models.Distance.EUCLID, models.Distance.MANHATTAN)
    ]
    + [("sparse", models.Distance.DOT)],
)
@pytest.mark.parametrize("limit", [1, 2, 5])
def test_mmr_distance_work_depends_on_result_limit(
    monkeypatch: pytest.MonkeyPatch, kind: str, distance: models.Distance, limit: int
) -> None:
    """Count real distance evaluations, without a timing-dependent assertion."""
    count = 24
    rng = np.random.default_rng(42)
    vectors = rng.random((count, 4)).tolist()
    query: Any = [1.0, 0.5, 0.25, 0.125]
    vectors_config = {"v": models.VectorParams(size=4, distance=distance)}
    sparse_vectors_config = None
    distance_function = "calculate_distance_core"
    if kind == "sparse":
        vectors_config = {}
        sparse_vectors_config = {"v": models.SparseVectorParams()}
        vectors = [models.SparseVector(indices=[0, 1, 2, 3], values=v) for v in vectors]
        query = models.SparseVector(indices=[0, 1, 2, 3], values=query)
        distance_function = "calculate_distance_sparse"
    elif kind == "multi":
        vectors_config["v"].multivector_config = models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        )
        vectors = [[v, [0.0, 1.0, 0.0, 0.0]] for v in vectors]
        query = [query]
        distance_function = "calculate_multi_distance_core"

    with closing(QdrantClient(":memory:")) as client:
        client.create_collection(
            "mmr",
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )
        client.upsert(
            "mmr",
            [models.PointStruct(id=i, vector={"v": v}) for i, v in enumerate(vectors)],
        )
        original = getattr(local_collection, distance_function)
        evaluations = 0

        def count_distances(
            query_vector: Any, candidate_vectors: Any, *args: Any, **kwargs: Any
        ) -> Any:
            nonlocal evaluations
            evaluations += len(candidate_vectors)
            return original(query_vector, candidate_vectors, *args, **kwargs)

        monkeypatch.setattr(local_collection, distance_function, count_distances)
        result = client.query_points(
            "mmr",
            query=models.NearestQuery(
                nearest=query, mmr=models.Mmr(diversity=0.5, candidates_limit=count)
            ),
            using="v",
            limit=limit,
        )

    assert len(result.points) == limit
    # Allow relevance/candidate search as well as a pass per selected point.
    # The eager N*N distance matrix exceeds this budget even for limit=1.
    assert evaluations <= count * (limit + 2)


@pytest.mark.parametrize(
    ("diversity", "expected"),
    [(0.0, [1, 2, 3, 4]), (0.5, [1, 2, 4, 3]), (1.0, [1, 4, 2, 3])],
)
def test_mmr_ties_and_diversity_endpoints(diversity: float, expected: list[int]) -> None:
    with closing(QdrantClient(":memory:")) as client:
        client.create_collection(
            "mmr", vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT)
        )
        client.upsert(
            "mmr",
            [
                models.PointStruct(id=1, vector=[2.0, 1.0]),
                models.PointStruct(id=2, vector=[1.0, 0.0]),
                models.PointStruct(id=3, vector=[0.5, 1.0]),
                models.PointStruct(id=4, vector=[0.25, 0.75]),
            ],
        )
        result = client.query_points(
            "mmr",
            query=models.NearestQuery(
                nearest=[1.0, 0.0], mmr=models.Mmr(diversity=diversity, candidates_limit=4)
            ),
            limit=10,
        )
    assert [point.id for point in result.points] == expected


def test_mmr_keeps_maximum_over_all_selected_points() -> None:
    with closing(QdrantClient(":memory:")) as client:
        client.create_collection(
            "mmr", vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT)
        )
        client.upsert(
            "mmr",
            [
                models.PointStruct(id=1, vector=[2.0, 0.0]),
                models.PointStruct(id=2, vector=[0.125, 2.0]),
                models.PointStruct(id=3, vector=[0.25, 2.0]),
                models.PointStruct(id=4, vector=[0.5, 0.0]),
                models.PointStruct(id=5, vector=[0.375, -2.0]),
            ],
        )
        result = client.query_points(
            "mmr",
            query=models.NearestQuery(nearest=[1.0, 0.0], mmr=models.Mmr(candidates_limit=5)),
            limit=5,
        )
    # After selecting 5, point 3 still has a large penalty from point 2.
    assert [point.id for point in result.points] == [1, 2, 5, 4, 3]


def test_mmr_multivector_penalty_is_candidate_to_selected() -> None:
    with closing(QdrantClient(":memory:")) as client:
        client.create_collection(
            "mmr",
            vectors_config=models.VectorParams(
                size=2,
                distance=models.Distance.DOT,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
        )
        client.upsert(
            "mmr",
            [
                models.PointStruct(id=1, vector=[[2.0, 1.0]]),
                models.PointStruct(id=2, vector=[[1.0, 0.0], [1.0, 0.0]]),
                models.PointStruct(id=3, vector=[[0.5, 1.0]]),
            ],
        )
        result = client.query_points(
            "mmr",
            query=models.NearestQuery(nearest=[[1.0, 0.0]], mmr=models.Mmr(candidates_limit=3)),
            limit=3,
        )
    # sim(2, 1)=4, whereas sim(1, 2)=2. Reversing it would select 2 before 3.
    assert [point.id for point in result.points] == [1, 3, 2]


@pytest.mark.parametrize("seed", [0, 26, 35, 46])
def test_mmr_cosine_preserves_eager_rounding_on_near_ties(seed: int) -> None:
    """Use an eager reference because near-tie rounding depends on NumPy's BLAS."""
    rng = np.random.default_rng(seed)
    vectors = (rng.integers(-2, 3, size=(15, 8)) / 4).tolist()
    query = vectors[0]
    collection = local_collection.LocalCollection(
        models.CreateCollection(
            vectors=models.VectorParams(size=8, distance=models.Distance.COSINE)
        )
    )
    collection.upsert([models.PointStruct(id=i, vector=v) for i, v in enumerate(vectors)])
    # Repeated cosine searches may change the last bits of stored normalization.
    for _ in range(13):
        candidates = collection.search(query, limit=15, with_vectors=True)

    raw_vectors = [np.array(point.vector) for point in candidates]
    targets = np.array(raw_vectors)
    pair_similarities = []
    relevance = []
    for vector in raw_vectors:
        pair_similarities.append(
            local_collection.calculate_distance_core(
                vector, targets, models.Distance.COSINE
            ).tolist()
        )
        relevance.append(
            local_collection.calculate_distance_core(
                np.array(query), vector[np.newaxis, :], models.Distance.COSINE
            ).tolist()[0]
        )

    # The reference scans the entire selected set, independently of the running maximum.
    pending = list(range(len(candidates)))
    selected: list[int] = []
    while pending:
        scores = {
            i: (
                relevance[i]
                if not selected
                else 0.5 * relevance[i] - 0.5 * max(pair_similarities[i][j] for j in selected)
            )
            for i in pending
        }
        best = max(reversed(pending), key=lambda i: scores[i])
        selected.append(best)
        position = pending.index(best)
        pending[position] = pending[-1]
        pending.pop()

    result = collection._mmr(candidates, query, "", lambda_=0.5, limit=15)
    assert result == [candidates[i] for i in selected]


@pytest.mark.parametrize("distance", list(models.Distance))
def test_mmr_pagination_filter_threshold_and_returned_fields(distance: models.Distance) -> None:
    with closing(QdrantClient(":memory:")) as client:
        client.create_collection(
            "mmr",
            vectors_config={
                "rank": models.VectorParams(size=2, distance=distance),
                "other": models.VectorParams(size=2, distance=models.Distance.DOT),
            },
        )
        client.upsert(
            "mmr",
            [
                models.PointStruct(
                    id=i,
                    vector={"rank": v, "other": [float(i), 1.0]},
                    payload={"keep": i != 4, "label": str(i)},
                )
                for i, v in enumerate(
                    [[1.0, 0.0], [0.75, 0.25], [0.0, 1.0], [-1.0, 0.0], [0.5, 0.5]]
                )
            ],
        )
        kwargs: dict[str, Any] = dict(
            using="rank",
            query_filter=models.Filter(
                must=[models.FieldCondition(key="keep", match=models.MatchValue(value=True))]
            ),
            score_threshold=(
                1.5 if distance in (models.Distance.EUCLID, models.Distance.MANHATTAN) else 0.0
            ),
            with_payload=["label"],
            with_vectors=["other"],
        )
        nearest = client.query_points("mmr", query=[1.0, 0.0], limit=10, **kwargs).points
        query = models.NearestQuery(nearest=[1.0, 0.0], mmr=models.Mmr(candidates_limit=10))
        full = client.query_points("mmr", query=query, limit=10, **kwargs).points
        page = client.query_points("mmr", query=query, limit=1, offset=1, **kwargs).points
        exhausted = client.query_points("mmr", query=query, limit=1, offset=10, **kwargs)

    assert len(full) >= 2
    assert page == full[1:2]
    assert exhausted.points == []
    assert {point.id: point for point in full} == {point.id: point for point in nearest}
    for point in full:
        assert isinstance(point.id, int)
        assert point.payload == {"label": str(point.id)}
        assert point.vector == {"other": [float(point.id), 1.0]}


@pytest.mark.asyncio
async def test_mmr_async_empty_and_single_candidate() -> None:
    client = AsyncQdrantClient(":memory:")
    try:
        await client.create_collection(
            "mmr", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE)
        )
        query = models.NearestQuery(nearest=[1.0, 0.0], mmr=models.Mmr())
        assert (await client.query_points("mmr", query=query)).points == []
        await client.upsert("mmr", [models.PointStruct(id=1, vector=[1.0, 0.0])])
        result = await client.query_points("mmr", query=query, limit=10)
    finally:
        await client.close()
    assert [point.id for point in result.points] == [1]
