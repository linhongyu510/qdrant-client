from qdrant_client import QdrantClient, models


def test_with_payload_include_nested_list():
    """Include pattern "a.b" should extract "b" from each dict in the "a" list."""
    client = QdrantClient(":memory:")
    client.create_collection(
        "test", vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT)
    )
    client.upsert(
        "test",
        points=[
            models.PointStruct(
                id=1,
                vector=[1.0, 0.0],
                payload={"a": [{"b": 1, "c": 2}, {"b": 3, "c": 4}], "d": 100},
            )
        ],
        wait=True,
    )

    records, _ = client.scroll("test", limit=1, with_payload=["a.b"])
    assert records[0].payload == {"a": [{"b": 1}, {"b": 3}]}

    records, _ = client.scroll("test", limit=1, with_payload=["a.c"])
    assert records[0].payload == {"a": [{"c": 2}, {"c": 4}]}


def test_with_payload_exclude_nested_list():
    """Exclude pattern "a.b" should remove "b" from each dict in the "a" list."""
    client = QdrantClient(":memory:")
    client.create_collection(
        "test", vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT)
    )
    client.upsert(
        "test",
        points=[
            models.PointStruct(
                id=1,
                vector=[1.0, 0.0],
                payload={"a": [{"b": 1, "c": 2}, {"b": 3, "c": 4}], "d": 100},
            )
        ],
        wait=True,
    )

    records, _ = client.scroll(
        "test", limit=1, with_payload=models.PayloadSelectorExclude(exclude=["a.b"])
    )
    assert records[0].payload == {"a": [{"c": 2}, {"c": 4}], "d": 100}

    records, _ = client.scroll(
        "test", limit=1, with_payload=models.PayloadSelectorExclude(exclude=["a.c"])
    )
    assert records[0].payload == {"a": [{"b": 1}, {"b": 3}], "d": 100}


def test_with_payload_include_whole_list_unchanged():
    """Include "a" (without subfield) should return the whole list."""
    client = QdrantClient(":memory:")
    client.create_collection(
        "test", vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT)
    )
    client.upsert(
        "test",
        points=[
            models.PointStruct(
                id=1,
                vector=[1.0, 0.0],
                payload={"a": [{"b": 1, "c": 2}, {"b": 3, "c": 4}], "d": 100},
            )
        ],
        wait=True,
    )

    records, _ = client.scroll("test", limit=1, with_payload=["a"])
    assert records[0].payload == {"a": [{"b": 1, "c": 2}, {"b": 3, "c": 4}]}
