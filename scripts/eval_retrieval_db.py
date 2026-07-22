import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import psycopg2
import torch
from pgvector.psycopg2 import register_vector
from transformers import AutoModel, AutoTokenizer

from src import config
from scripts.embed_addresses import embed_batch, get_device

ANCHOR_FILE = "../data/address/STT발화_정답_학습데이터.txt"

BATCH_SIZE = 64
TOPKS = (1, 5, 10)

MODELS = [
    {"name": "bge-m3-korean", "path": config.MODEL_NAME, "table": "train_korean_20260720"},
    {"name": "bge-m3-address", "path": config.OUTPUT_DIR, "table": "train_address_20260720"},
]


def get_connection():
    conn = psycopg2.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
    )
    register_vector(conn)
    return conn


def load_pairs():
    queries = []
    answers = []
    with open(ANCHOR_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            q, a = line.split("\t")
            queries.append(q)
            answers.append(a)
    return queries, answers


def embed_all(tokenizer, model, device, texts):
    vecs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        vecs.append(embed_batch(tokenizer, model, device, batch))
    return np.concatenate(vecs, axis=0)


def evaluate(model_conf, queries, answers, device):
    name = model_conf["name"]
    path = model_conf["path"]
    table = model_conf["table"]

    print(f"\n=== {name} (table={table}) ===")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path)
    model.to(device)
    model.eval()

    with torch.no_grad():
        query_vecs = embed_all(tokenizer, model, device, queries)

    conn = get_connection()
    cur = conn.cursor()

    ranks = []
    not_found = 0
    rows = []
    start = time.time()

    for i, (q, ans, qvec) in enumerate(zip(queries, answers, query_vecs)):
        # 정답 주소가 DB(gallery)에 실제로 존재하는지, 존재한다면 query와의 거리
        cur.execute(
            f"SELECT embedding <=> %s AS dist FROM {table} WHERE address = %s LIMIT 1;",
            (qvec, ans),
        )
        result = cur.fetchone()

        # top-1 예측
        cur.execute(
            f"SELECT address, embedding <=> %s AS dist FROM {table} ORDER BY embedding <=> %s LIMIT 1;",
            (qvec, qvec),
        )
        top1_address, top1_dist = cur.fetchone()

        if result is None:
            not_found += 1
            rows.append({
                "query": q,
                "answer": ans,
                "found_in_db": False,
                "rank": "",
                "top1_prediction": top1_address,
                "top1_similarity": 1 - top1_dist,
                "true_similarity": "",
            })
            continue

        true_dist = result[0]

        # 정답보다 query에 더 가까운(=더 낮은 distance) 행 개수 + 1 = rank
        cur.execute(
            f"SELECT count(*) FROM {table} WHERE embedding <=> %s < %s;",
            (qvec, true_dist),
        )
        rank = cur.fetchone()[0] + 1
        ranks.append(rank)

        rows.append({
            "query": q,
            "answer": ans,
            "found_in_db": True,
            "rank": rank,
            "top1_prediction": top1_address,
            "top1_similarity": 1 - top1_dist,
            "true_similarity": 1 - true_dist,
        })

        done = i + 1
        if done % 50 == 0 or done == len(queries):
            elapsed = time.time() - start
            print(f"[{name}] {done}/{len(queries)} ({elapsed:.1f}s elapsed)")

    cur.close()
    conn.close()

    ranks_arr = np.array(ranks) if ranks else np.array([])
    n_found = len(ranks_arr)

    metrics = {f"recall@{k}": float((ranks_arr <= k).mean()) if n_found else 0.0 for k in TOPKS}
    metrics["mrr"] = float((1.0 / ranks_arr).mean()) if n_found else 0.0
    metrics["mean_rank"] = float(ranks_arr.mean()) if n_found else 0.0
    metrics["not_found_in_db"] = not_found
    metrics["n_total"] = len(queries)

    print(f"n_total={len(queries)}  not_found_in_db={not_found}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    out_csv = f"../data/eval_db_{name}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"per-query results -> {out_csv}")

    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    return metrics


def main():
    device = get_device()
    print(f"Using device: {device}")

    queries, answers = load_pairs()
    print(f"queries: {len(queries)}")

    all_metrics = {}
    for model_conf in MODELS:
        all_metrics[model_conf["name"]] = evaluate(model_conf, queries, answers, device)

    print("\n=== Summary ===")
    header = ["model"] + list(next(iter(all_metrics.values())).keys())
    print("\t".join(header))
    for name, m in all_metrics.items():
        vals = [f"{m[k]:.4f}" if isinstance(m[k], float) else str(m[k]) for k in header[1:]]
        print("\t".join([name] + vals))


if __name__ == "__main__":
    main()
