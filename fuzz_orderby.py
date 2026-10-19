"""Fuzz local order_by scroll: compare collected distinct points to ground truth."""
import random

from qdrant_client import QdrantClient, models
from qdrant_client.local.order_by import to_order_value


def scroll_all(client, key, limit, direction, points_meta):
    collected = []
    last_seen_value = None
    last_value_ids = []
    pages = 0
    while True:
        order_by = models.OrderBy(key=key, start_from=last_seen_value, direction=direction)
        records, next_page = client.scroll(
            collection_name="col",
            limit=limit,
            order_by=order_by,
            scroll_filter=models.Filter(must_not=[models.HasIdCondition(has_id=last_value_ids)]),
            with_payload=True,
        )
        pages += 1
        if pages > 500:
            raise AssertionError("infinite loop")
        if len(records) == 0:
            break
        last_value = records[-1].order_value
        if last_seen_value != last_value:
            last_seen_value = last_value
            last_value_ids = []
        for record in records:
            if isinstance(record.payload[key], list):
                order_value_payload = [to_order_value(item) for item in record.payload[key]]
                if last_seen_value in order_value_payload:
                    last_value_ids.append(record.id)
            else:
                order_value_payload = to_order_value(record.payload[key])
                if last_seen_value == order_value_payload:
                    last_value_ids.append(record.id)
        collected.extend(records)
    return collected


def run(seed):
    rng = random.Random(seed)
    n = rng.randint(3, 40)
    points = []
    truth = set()
    for i in range(n):
        pid = i + 1
        # value: either scalar int or a small array of ints
        if rng.random() < 0.3:
            vals = sorted(rng.sample(range(0, 8), rng.randint(1, 3)))
            payload = {"rank": vals}
        else:
            v = rng.randint(0, 7)
            payload = {"rank": v}
        points.append(models.PointStruct(id=pid, vector=[float(i % 3)], payload=payload))
        truth.add(pid)

    client = QdrantClient(":memory:")
    client.create_collection("col", vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE))
    client.upsert("col", points)

    for direction in [models.Direction.ASC, models.Direction.DESC]:
        for limit in [1, 2, 3, 5, 10]:
            collected = scroll_all(client, "rank", limit, direction, points)
            got = {r.id for r in collected}
            missing = truth - got
            extra = got - truth
            if missing or extra:
                print(f"seed={seed} dir={direction} limit={limit}: MISSING={sorted(missing)} EXTRA={sorted(extra)}")
                # dump points
                for p in points:
                    print("   ", p.id, p.payload)
                return False
    return True


def main():
    bad = 0
    for seed in range(300):
        if not run(seed):
            bad += 1
            if bad > 5:
                break
    print(f"done. bad_seeds={bad}")


if __name__ == "__main__":
    main()
