"""Compare local vs remote order_by scroll, page by page."""
import sys

from qdrant_client import QdrantClient, models
from qdrant_client.local.order_by import to_order_value

COL = "cmp_col"


def setup(client):
    if client.collection_exists(COL):
        client.delete_collection(COL)
    client.create_collection(COL, vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE))
    points = []
    for i in range(30):
        pid = i + 1
        points.append(models.PointStruct(id=pid, vector=[float(i % 3)], payload={"rank": (i % 6) + 1}))
    client.upsert(COL, points)
    client.create_payload_index(COL, "rank", models.PayloadSchemaType.INTEGER, wait=True)


def paginate(client, limit, direction):
    """Naive pagination: start_from = last value, no must_not bookkeeping."""
    pages = []
    last_seen = None
    last_value_ids = []
    seen_ids_cum = set()
    for step in range(100):
        order_by = models.OrderBy(key="rank", start_from=last_seen, direction=direction)
        records, next_page = client.scroll(
            collection_name=COL,
            limit=limit,
            order_by=order_by,
            scroll_filter=models.Filter(must_not=[models.HasIdCondition(has_id=last_value_ids)]),
            with_payload=True,
        )
        pages.append((len(records), next_page, [(r.id, r.order_value) for r in records]))
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
    return pages


def main():
    local = QdrantClient(":memory:")
    remote = QdrantClient(host="localhost", port=6333)
    setup(local)
    setup(remote)

    for direction in [models.Direction.ASC, models.Direction.DESC]:
        for limit in [3, 5, 7]:
            lp = paginate(local, limit, direction)
            rp = paginate(remote, limit, direction)
            same = lp == rp
            print(f"dir={direction} limit={limit}: local_pages={len(lp)} remote_pages={len(rp)} identical={same}")
            if not same:
                # show first divergence
                for i, (a, b) in enumerate(zip(lp, rp)):
                    if a != b:
                        print(f"  page {i} DIVERGES:")
                        print(f"    local : {a}")
                        print(f"    remote: {b}")
                        break
                if len(lp) != len(rp):
                    print(f"  length diff: local={len(lp)} remote={len(rp)}")


if __name__ == "__main__":
    main()
