import time

import psycopg2
import torch
import torch.nn.functional as F
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values
from transformers import AutoModel, AutoTokenizer

from src import config

BATCH_SIZE = 64

MODELS = [
    {"name": "bge-m3-korean", "path": config.MODEL_NAME, "table": "train_korean"},
    {"name": "bge-m3-address", "path": config.OUTPUT_DIR, "table": "train_address"},
]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_addresses(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


@torch.no_grad()
def embed_batch(tokenizer, model, device, texts):
    encoded = tokenizer(
        texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
    ).to(device)

    output = model(**encoded)
    token_embeddings = output.last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1).float()
    summed = (token_embeddings * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    embeddings = summed / counts
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()


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


def embed_and_store(model_conf, addresses, device):
    name = model_conf["name"]
    path = model_conf["path"]
    table = model_conf["table"]

    print(f"\n=== {name} -> {table} ===")
    print(f"Loading model from {path} ...")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path)
    model.to(device)
    model.eval()

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY;")
    conn.commit()

    total = len(addresses)
    start = time.time()

    for i in range(0, total, BATCH_SIZE):
        batch = addresses[i:i + BATCH_SIZE]
        embeddings = embed_batch(tokenizer, model, device, batch)

        rows = [(addr, vec) for addr, vec in zip(batch, embeddings)]
        execute_values(
            cur,
            f"INSERT INTO {table} (address, embedding) VALUES %s",
            rows,
            template="(%s, %s)",
        )
        conn.commit()

        done = min(i + BATCH_SIZE, total)
        if done % (BATCH_SIZE * 20) == 0 or done == total:
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(f"[{name}] {done}/{total} ({rate:.1f} addr/s, ETA {eta/60:.1f}m)")

    cur.close()
    conn.close()

    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    print(f"[{name}] done in {(time.time() - start)/60:.1f}m")


def main():
    device = get_device()
    print(f"Using device: {device}")

    addresses = load_addresses(config.LEARN_ADDRESS_FILE)
    print(f"Loaded {len(addresses)} addresses")

    for model_conf in MODELS:
        embed_and_store(model_conf, addresses, device)


if __name__ == "__main__":
    main()
