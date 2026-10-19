"""Reproduce local-mode order_by scroll pagination behavior.

Ground truth: every distinct (value, point_id) pair for points that carry the
order-by key, sorted by value. We then paginate exactly like the congruence test
(`order_by.start_from` + `must_not has_id` at the boundary) and check whether the
union of pages equals the ground truth.
"""
from qdrant_client import QdrantClient, models
from qdrant_client.conversions.common_types import Record


def build_client(n: int, key: str = "rank") -> QdrantClient:
    client = QdrantClient(":memory:")
    client.create_collection(
        "col",
        vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
    )
    points = []
    for i in range(n):
        # values 1..5 repeated => heavy ties; order_by key = rank
        points.append(
            models.PointStruct(
                id=i + 1,
                vector=[float(i % 3)],
                payload={key: (i % 5) + 1},
            )
        )
    client.upsert("col", points)
    return client


def scroll_all(client: QdrantClient, key: str, limit: int):
    """Mirror tests/congruence_tests/test_scroll_order_by.py scroll_all_with_key."""
    collected: list[Record] = []
    last_seen_value = None
    last_value_ids: list = []
    pages = 0
    while True:
        start_from = last_seen_value
        records, next_page = client.scroll(
            collection_name="col",
            limit=limit,
            order_by=models.OrderBy(key=key, start_from=start_from),
            scroll_filter=models.Filter(must_not=[models.HasIdCondition(has_id=last_value_ids)]),
            with_payload=True,
        )
        pages += 1
        if pages > 1000:
            raise AssertionError("too many pages, likely infinite loop")
        if len(records) == 0:
            break
        last_value = records[-1].order_value
        if last_seen_value != last_value:
            last_seen_value = last_value
            last_value_ids = []
        for record in records:
            v = record.payload[key]
            if last_seen_value == v:
                last_value_ids.append(record.id)
        collected.extend(records)
    return collected


def main() -> None:
    for n, limit in [(20, 6), (50, 10), (100, 20)]:
        client = build_client(n)
        # ground truth: distinct point ids that have the key
        all_records, _ = client.scroll("col", limit=100000, with_payload=True)
        truth_ids = sorted(r.id for r in all_records)
        collected = scroll_all(client, "rank", limit)
        got_ids = sorted({r.id for r in collected})
        print(f"n={n} limit={limit}: truth_ids={len(truth_ids)} collected_distinct_ids={len(got_ids)}")
        missing = set(truth_ids) - set(got_ids)
        extra = set(got_ids) - set(truth_ids)
        if missing:
            print(f"  !! MISSING (truncated) ids: {sorted(missing)}")
        if extra:
            print(f"  !! EXTRA ids: {sorted(extra)}")
        if not missing and not extra:
            print("  OK: full coverage")


if __name__ == "__main__":
    main()
