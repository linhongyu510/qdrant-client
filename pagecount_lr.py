"""Page-by-page count comparison local vs remote under congruence pagination."""
import random

from qdrant_client import QdrantClient, models
from qdrant_client.local.order_by import to_order_value

COL = "pp_col"


def paginate(client, limit, direction):
    pages = []
    last_seen = None
    last_value_ids = []
    for _ in range(500):
        order_by = models.OrderBy(key="rank", start_from=last_seen, direction=direction)
        records, _ = client.scroll(
            collection_name=COL, limit=limit, order_by=order_by,
            scroll_filter=models.Filter(must_not=[models.HasIdCondition(has_id=last_value_ids)]),
            with_payload=True,
        )
        pages.append(len(records))
        if len(records) == 0:
            break
        last_value = records[-1].order_value
        if last_seen != last_value:
            last_seen = last_value
            last_value_ids = []
        for r in records:
            v = r.payload["rank"]
            items = v if isinstance(v, list) else [v]
            if last_seen in [to_order_value(x) for x in items]:
                last_value_ids.append(r.id)
    return pages


def make_points(seed):
    rng = random.Random(seed)
    return [
        models.PointStruct(id=i + 1, vector=[float(i % 3)], payload={"rank": rng.randint(0, 5)})
        for i in range(rng.randint(5, 40))
    ]


def main():
    local = QdrantClient(":memory:")
    remote = QdrantClient(host="localhost", port=6333)
    for seed in range(40):
        points = make_points(seed)
        for c in (local, remote):
            if c.collection_exists(COL):
                c.delete_collection(COL)
            c.create_collection(COL, vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE))
            c.upsert(COL, points)
            c.create_payload_index(COL, "rank", models.PayloadSchemaType.INTEGER, wait=True)
        for direction in [models.Direction.ASC, models.Direction.DESC]:
            for limit in [1, 2, 3, 5, 8]:
                lp = paginate(local, limit, direction)
                rp = paginate(remote, limit, direction)
                if lp != rp:
                    print(f"seed={seed} dir={direction} limit={limit}: local_pages={lp} remote_pages={rp}")
    print("done")


if __name__ == "__main__":
    main()
