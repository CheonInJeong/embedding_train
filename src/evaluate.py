from sentence_transformers import SentenceTransformer
import torch

# ===========================
# 설정
# ===========================

MODEL_BEFORE = "../models/bge-m3-korean"
MODEL_AFTER = "../models/bge-m3-address"

ADDRESS_PATH = "../data/부산.txt"
VALID_PATH = "../data/valid.tsv"

TOP_K = 10


# ===========================
# 데이터 로드
# ===========================

def load_addresses(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_valid(path):
    dataset = []

    with open(path, encoding="utf-8") as f:
        for line in f:

            line = line.strip()

            if not line:
                continue

            anchor, positive = line.split("\t")

            dataset.append((anchor, positive))

    return dataset


# ===========================
# 주소 임베딩
# ===========================

def build_embeddings(model, addresses):

    print("Embedding address database...")

    embeddings = model.encode(
        addresses,
        batch_size=64,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings


# ===========================
# 검색
# ===========================

def search(model, db_embeddings, addresses, query, topk):

    query_embedding = model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    scores = torch.matmul(
        db_embeddings,
        query_embedding
    )

    values, indices = torch.topk(scores, k=topk)

    results = []

    for score, idx in zip(values.tolist(), indices.tolist()):

        results.append(
            (
                addresses[idx],
                score
            )
        )

    return results


# ===========================
# 평가
# ===========================

def evaluate(model, db_embeddings, addresses, valid):

    top1_correct = 0

    for idx, (anchor, positive) in enumerate(valid, 1):

        results = search(
            model,
            db_embeddings,
            addresses,
            anchor,
            TOP_K
        )

        print("=" * 80)
        print(f"[{idx}]")
        print("Anchor :", anchor)
        print("Answer :", positive)
        print()

        found = False

        for rank, (address, score) in enumerate(results, 1):

            mark = ""

            if address == positive:
                mark = " <-- 정답"

                if rank == 1:
                    top1_correct += 1

                found = True

            print(
                f"{rank:2d}. {score:.4f} {address}{mark}"
            )

        if not found:
            print("정답이 Top10에 없습니다.")

        print()

    accuracy = top1_correct / len(valid)

    return accuracy


# ===========================
# Main
# ===========================

def main():

    print("Loading models...")

    before_model = SentenceTransformer(MODEL_BEFORE)
    after_model = SentenceTransformer(MODEL_AFTER)

    print("Loading addresses...")

    addresses = load_addresses(ADDRESS_PATH)

    valid = load_valid(VALID_PATH)

    print()

    print("Building Before Embeddings")

    before_db = build_embeddings(
        before_model,
        addresses
    )

    print()

    print("Building After Embeddings")

    after_db = build_embeddings(
        after_model,
        addresses
    )

    print()
    print("========== BEFORE ==========")

    before_acc = evaluate(
        before_model,
        before_db,
        addresses,
        valid
    )

    print()
    print("========== AFTER ==========")

    after_acc = evaluate(
        after_model,
        after_db,
        addresses,
        valid
    )

    print()
    print("=" * 80)

    print(f"Top1 Accuracy Before : {before_acc:.2%}")
    print(f"Top1 Accuracy After  : {after_acc:.2%}")
    print(f"Improvement          : {(after_acc-before_acc):.2%}")


if __name__ == "__main__":
    main()