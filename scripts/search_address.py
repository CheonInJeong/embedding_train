import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg2
from pgvector.psycopg2 import register_vector
from transformers import AutoModel, AutoTokenizer

from src import config
from scripts.embed_addresses import embed_batch, get_device

TABLE = "train_address"
TOP_K = 5


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


def search(text, top_k=TOP_K):
    device = get_device()
    print(f"Using device: {device}")

    print(f"Loading model from {config.OUTPUT_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(config.OUTPUT_DIR)
    model = AutoModel.from_pretrained(config.OUTPUT_DIR)
    model.to(device)
    model.eval()

    query_vec = embed_batch(tokenizer, model, device, [text])[0]

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT address, embedding <=> %s AS distance
        FROM {TABLE}
        ORDER BY embedding <=> %s
        LIMIT %s;
        """,
        (query_vec, query_vec, top_k),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    print(f"\nQuery: {text}")
    for rank, (address, distance) in enumerate(rows, start=1):
        similarity = 1 - distance  # 정규화된 벡터 기준 cosine distance -> similarity
        print(f"  {rank}. {address}  (similarity={similarity:.4f})")


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "서울시 강남구 테헤란로"
    search(text)


if __name__ == "__main__":
    main()
