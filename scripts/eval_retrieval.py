import csv

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src import config
from scripts.embed_addresses import embed_batch, get_device

QUERY_FILE = "data/STT발화"
ANSWER_FILE = "data/STT발화_정답"

BATCH_SIZE = 64
TOPKS = (1, 5, 10)

MODELS = [
    {"name": "bge-m3-korean", "path": config.MODEL_NAME},
    {"name": "bge-m3-address", "path": config.OUTPUT_DIR},
]


def load_pairs():
    with open(QUERY_FILE, encoding="utf-8") as f:
        queries = [l.strip() for l in f if l.strip()]
    with open(ANSWER_FILE, encoding="utf-8") as f:
        answers = [l.strip() for l in f if l.strip()]
    assert len(queries) == len(answers)
    return queries, answers


def build_gallery(answers):
    gallery = []
    index = {}
    for a in answers:
        if a not in index:
            index[a] = len(gallery)
            gallery.append(a)
    return gallery, index


def embed_all(tokenizer, model, device, texts):
    vecs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        vecs.append(embed_batch(tokenizer, model, device, batch))
    return np.concatenate(vecs, axis=0)


def evaluate(model_conf, queries, answers, gallery, gallery_index, device):
    name = model_conf["name"]
    path = model_conf["path"]

    print(f"\n=== {name} ===")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path)
    model.to(device)
    model.eval()

    with torch.no_grad():
        query_vecs = embed_all(tokenizer, model, device, queries)
        gallery_vecs = embed_all(tokenizer, model, device, gallery)

    # normalized vectors -> dot product == cosine similarity
    sims = query_vecs @ gallery_vecs.T  # [num_queries, num_gallery]

    ranks = []
    rows = []
    for i, ans in enumerate(answers):
        true_idx = gallery_index[ans]
        order = np.argsort(-sims[i])
        rank = int(np.where(order == true_idx)[0][0]) + 1
        ranks.append(rank)

        top1_idx = order[0]
        rows.append({
            "query": queries[i],
            "answer": ans,
            "rank": rank,
            "top1_prediction": gallery[top1_idx],
            "top1_similarity": float(sims[i, top1_idx]),
            "true_similarity": float(sims[i, true_idx]),
        })

    ranks = np.array(ranks)
    n = len(ranks)

    metrics = {f"recall@{k}": float((ranks <= k).mean()) for k in TOPKS}
    metrics["mrr"] = float((1.0 / ranks).mean())
    metrics["mean_rank"] = float(ranks.mean())

    print(f"n={n}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    out_csv = f"data/eval_{name}.csv"
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
    gallery, gallery_index = build_gallery(answers)
    print(f"queries: {len(queries)}, unique gallery addresses: {len(gallery)}")

    all_metrics = {}
    for model_conf in MODELS:
        all_metrics[model_conf["name"]] = evaluate(
            model_conf, queries, answers, gallery, gallery_index, device
        )

    print("\n=== Summary ===")
    header = ["model"] + list(next(iter(all_metrics.values())).keys())
    print("\t".join(header))
    for name, m in all_metrics.items():
        print("\t".join([name] + [f"{m[k]:.4f}" for k in header[1:]]))


if __name__ == "__main__":
    main()
