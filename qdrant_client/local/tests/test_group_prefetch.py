from copy import deepcopy
from inspect import isawaitable
from typing import Awaitable, TypeVar

import pytest

from qdrant_client import AsyncQdrantClient, QdrantClient, models

T = TypeVar("T")


async def resolve(value: T | Awaitable[T]) -> T:
    return await value if isawaitable(value) else value


@pytest.mark.asyncio
@pytest.mark.parametrize("client_type", [QdrantClient, AsyncQdrantClient], ids=["sync", "async"])
@pytest.mark.parametrize(
    "prefetch, expected",
    [
        pytest.param(
            models.Prefetch(query=0, using="dense", limit=5),
            [("a", [1, 2]), ("b", [3])],
            id="bare-id",
        ),
        pytest.param(
            models.Prefetch(query=models.NearestQuery(nearest=0), using="dense", limit=5),
            [("a", [1, 2]), ("b", [3])],
            id="nearest-id",
        ),
        pytest.param(
            [models.Prefetch(query=0, using="dense", limit=5)],
            [("a", [1, 2]), ("b", [3])],
            id="list",
        ),
        pytest.param(
            models.Prefetch(
                prefetch=models.Prefetch(query=0, using="dense", limit=5),
                query=0,
                using="dense",
                score_threshold=0.75,
                limit=5,
            ),
            [("a", [1, 2])],
            id="nested",
        ),
        pytest.param(
            models.Prefetch(
                query=models.RecommendQuery(
                    recommend=models.RecommendInput(positive=[0], negative=[4])
                ),
                using="dense",
                limit=5,
            ),
            [("a", [1, 2]), ("b", [3])],
            id="recommend-ids",
        ),
        pytest.param(
            models.Prefetch(
                query=0,
                using="dense",
                lookup_from=models.LookupLocation(collection="examples", vector="seed"),
                limit=5,
            ),
            [("b", [3]), ("a", [2, 1]), ("source", [0])],
            id="foreign-collection",
        ),
        pytest.param(
            models.Prefetch(
                query=models.RecommendQuery(recommend=models.RecommendInput(positive=[0])),
                using="dense",
                lookup_from=models.LookupLocation(collection="examples", vector="seed"),
                limit=5,
            ),
            [("b", [3]), ("a", [2, 1]), ("source", [0])],
            id="foreign-recommend",
        ),
    ],
)
async def test_group_prefetch_resolves_ids(
    client_type: type[QdrantClient] | type[AsyncQdrantClient],
    prefetch: models.Prefetch | list[models.Prefetch],
    expected: list[tuple[str, list[int]]],
) -> None:
    client = client_type(":memory:")
    try:
        await resolve(
            client.create_collection(
                "documents",
                vectors_config={
                    "dense": models.VectorParams(size=2, distance=models.Distance.DOT)
                },
            )
        )
        await resolve(
            client.upsert(
                "documents",
                [
                    models.PointStruct(
                        id=point_id,
                        vector={"dense": vector},
                        payload={"document": document, "keep": point_id != 4},
                    )
                    for point_id, vector, document in [
                        (0, [1.0, 0.0], "source"),
                        (1, [0.9, 0.1], "a"),
                        (2, [0.8, 0.2], "a"),
                        (3, [0.7, 0.3], "b"),
                        (4, [0.0, 1.0], "excluded"),
                    ]
                ],
                wait=True,
            )
        )
        await resolve(
            client.create_collection(
                "examples",
                vectors_config={"seed": models.VectorParams(size=2, distance=models.Distance.DOT)},
            )
        )
        await resolve(
            client.upsert(
                "examples",
                [models.PointStruct(id=0, vector={"seed": [0.0, 1.0]})],
                wait=True,
            )
        )

        prefetch = deepcopy(prefetch)
        for source in prefetch if isinstance(prefetch, list) else [prefetch]:
            source.filter = models.Filter(
                must=[models.FieldCondition(key="keep", match=models.MatchValue(value=True))]
            )
        original = deepcopy(prefetch)

        result = await resolve(
            client.query_points_groups(
                "documents",
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                prefetch=prefetch,
                group_by="document",
                group_size=2,
                limit=10,
            )
        )

        assert [(group.id, [hit.id for hit in group.hits]) for group in result.groups] == expected
        assert prefetch == original
    finally:
        await resolve(client.close())
