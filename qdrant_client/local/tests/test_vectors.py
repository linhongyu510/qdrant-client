import random

from qdrant_client import models
from qdrant_client.local.local_collection import DEFAULT_VECTOR_NAME, LocalCollection


def test_get_vectors():
    collection = LocalCollection(
        models.CreateCollection(
            vectors=models.VectorParams(size=2, distance=models.Distance.MANHATTAN)
        )
    )
    collection.upsert(
        points=[
            models.PointStruct(id=i, vector=[random.random(), random.random()]) for i in range(10)
        ]
    )

    assert collection._get_vectors(idx=1, with_vectors=DEFAULT_VECTOR_NAME)
    assert collection._get_vectors(idx=2, with_vectors=True)
    assert collection._get_vectors(idx=3, with_vectors=False) is None


def test_reupsert_identical_cosine_vector_is_idempotent():
    collection = LocalCollection(
        models.CreateCollection(
            vectors={"dense": models.VectorParams(size=4, distance=models.Distance.COSINE)}
        )
    )
    point = models.PointStruct(
        id=1,
        vector={"dense": [0.1234567901234, -0.98765432109, 0.5555555555, 0.333333333333]},
    )

    collection.upsert(points=[point])
    first = collection.vectors["dense"][0].tolist()
    collection.upsert(points=[point])
    second = collection.vectors["dense"][0].tolist()

    assert second == first
