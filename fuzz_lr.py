"""Fuzz local vs remote order_by scroll: compare the COLLECTED SETS, not page order."""
import random

from qdrant_client import QdrantClient, models
from qdrant_client.local.order_by import to_order_value

COL = "fuzz_col"


def setup(client, points):
    if client.collection_exists(COL):
        client.delete_collection(COL)
    client.create_collection(COL, vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE))
    client.upsert(COL, points)
    client.create_payload_index(COL, "rank", models.PayloadSchemaType.INTEGER, wait=True)


def paginate(client, limit, direction):
    """Congruence-style pagination; return sorted list of (value, id)."""
    out = []
    last_seen = None
    last_value_ids = []
    for step in range(500):
        order_by = models.OrderBy(key="rank", start_from=last_seen, direction=direction)
        records, _ = client.scroll(
            collection_name=COL,
            limit=limit,
            order_by=order_by,
            scroll_filter=models.Filter(must_not=[models.HasIdCondition(has_id=last_value_ids)]),
            with_payload=True,
        )
        if len(records) == 0:
            break
        last_value = records[-1].order_value
        if last_seen != last_value:
            last_seen = last_value
            last_value_ids = []
        for r in records:
            v = r.payload["rank"]
            if last_seen == v:
                last_value_ids.append(r.id)
        for r in records:
            out.append((r.order_value, r.id))
    return sorted(out)


def make_points(seed):
    rng = random.Random(seed)
    points = []
    for i in range(rng.randint(3, 40)):
        pid = i + 1
        points.append(
            models.PointStruct(id=pid, vector=[float(i % 3)], payload={"rank": rng.randint(0, 5)})
        )
    return points


def main():
    local = QdrantClient(":memory:")
    remote = QdrantClient(host="localhost", port=6333)
    bad = 0
    for seed in range(60):
        points = make_points(seed)
        setup(local, points)
        setup(remote, points)
        for direction in [models.Direction.ASC, models.Direction.DESC]:
            for limit in [1, 2, 3, 5, 8]:
                ls = paginate(local, limit, direction)
                rs = paginate(remote, limit, direction)
                if ls != rs:
                    bad += 1
                    print(f"seed={seed} dir={direction} limit={limit} DIVERGES")
                    lset, rset = set(ls), set(rs)
                    print(f"  local-only: {sorted(lset - rset)}")
                    print(f"  remote-only: {sorted(rset - lset)}")
                    if bad > 8:
                        print("too many, stopping")
                        return
    print(f"done. bad={bad}")


if __name__ == "__main__":
    main()
