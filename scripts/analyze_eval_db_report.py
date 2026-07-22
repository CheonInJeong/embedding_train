import csv

import numpy as np

# 사용법##########################################################################################
# python -m scripts.eval_retrieval_db  # 1) DB 대상 쿼리별 결과 생성 (data/eval_db_bge-m3-*.csv)   #
# python -m scripts.analyze_eval_db_report  # 2) 그 결과로 지표/케이스분해/사례 출력 + CSV 저장       #
#################################################################################################

# scripts/eval_retrieval_db.py가 만든 두 CSV(모델별 쿼리 결과)를 비교 분석해서
# docs/eval_report_db.md 작성에 필요한 지표/케이스/사례를 뽑아낸다.

KOREAN_CSV = "data/eval_db_bge-m3-korean.csv"
ADDRESS_CSV = "data/eval_db_bge-m3-address.csv"

KOREAN_ONLY_CSV = "data/eval_db_korean_only_correct.csv"
ADDRESS_ONLY_CSV = "data/eval_db_address_only_correct.csv"
BOTH_WRONG_CSV = "data/eval_db_both_wrong.csv"

N_EXAMPLES = 8


def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rank(row):
    if row["found_in_db"] != "True" or row["rank"] == "":
        return None
    return int(row["rank"])


def print_metrics(name, rows):
    ranks = np.array([r for r in (rank(row) for row in rows) if r is not None])
    n_total = len(rows)
    n_found = len(ranks)

    print(f"--- {name} ---")
    print(f"total={n_total}  found_in_db={n_found}  not_found={n_total - n_found}")
    for k in (1, 5, 10):
        print(f"  recall@{k} (found 기준): {(ranks <= k).mean():.4f}")
    print(f"  mrr (found 기준): {(1.0 / ranks).mean():.4f}")
    print(f"  mean_rank: {ranks.mean():.2f}  median_rank: {np.median(ranks):.1f}")
    print(f"  rank<=2 (근접 오답 포함): {(ranks <= 2).mean():.4f}")
    print(f"  rank>100: {(ranks > 100).mean():.4f} ({(ranks > 100).sum()}건)")
    print(f"  rank>1000: {(ranks > 1000).mean():.4f} ({(ranks > 1000).sum()}건)")
    print()


def main():
    korean_rows = load(KOREAN_CSV)
    address_rows = load(ADDRESS_CSV)
    assert len(korean_rows) == len(address_rows)

    print_metrics("bge-m3-korean", korean_rows)
    print_metrics("bge-m3-address", address_rows)

    k_not_found = set(r["answer"] for r in korean_rows if r["found_in_db"] != "True")
    a_not_found = set(r["answer"] for r in address_rows if r["found_in_db"] != "True")
    print(f"not_found 주소 집합 동일 여부: {k_not_found == a_not_found} "
          f"(korean {len(k_not_found)}개, address {len(a_not_found)}개)\n")

    merged = []
    for k, a in zip(korean_rows, address_rows):
        assert k["query"] == a["query"]
        merged.append({
            "query": k["query"],
            "answer": k["answer"],
            "k_found": k["found_in_db"] == "True",
            "k_rank": rank(k),
            "k_top1": k["top1_prediction"],
            "a_found": a["found_in_db"] == "True",
            "a_rank": rank(a),
            "a_top1": a["top1_prediction"],
        })

    both_found = [m for m in merged if m["k_found"] and m["a_found"]]

    def correct(r):
        return r == 1

    n_both_correct = sum(1 for m in both_found if correct(m["k_rank"]) and correct(m["a_rank"]))
    n_both_wrong = sum(1 for m in both_found if not correct(m["k_rank"]) and not correct(m["a_rank"]))
    n_korean_only = sum(1 for m in both_found if correct(m["k_rank"]) and not correct(m["a_rank"]))
    n_address_only = sum(1 for m in both_found if correct(m["a_rank"]) and not correct(m["k_rank"]))

    n_improved = sum(1 for m in both_found if m["k_rank"] > m["a_rank"])
    n_worse = sum(1 for m in both_found if m["k_rank"] < m["a_rank"])
    n_same = sum(1 for m in both_found if m["k_rank"] == m["a_rank"])

    print("=== 케이스 분해 (both found_in_db 기준, "
          f"{len(both_found)}건) ===")
    print(f"both correct: {n_both_correct}")
    print(f"both wrong: {n_both_wrong}")
    print(f"korean only correct: {n_korean_only}")
    print(f"address only correct: {n_address_only}")
    print(f"rank improved(korean->address): {n_improved}  worse: {n_worse}  same: {n_same}")
    print()

    korean_only_examples = [m for m in both_found if correct(m["k_rank"]) and not correct(m["a_rank"])]
    address_only_examples = [m for m in both_found if correct(m["a_rank"]) and not correct(m["k_rank"])]
    both_wrong_examples = sorted(
        [m for m in both_found if not correct(m["k_rank"]) and not correct(m["a_rank"])],
        key=lambda m: min(m["k_rank"], m["a_rank"]),
    )

    def save(path, rows):
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved -> {path} ({len(rows)}건)")

    save(KOREAN_ONLY_CSV, korean_only_examples)
    save(ADDRESS_ONLY_CSV, address_only_examples)
    save(BOTH_WRONG_CSV, both_wrong_examples)

    print()
    print(f"=== address_only 사례 (상위 {N_EXAMPLES}건) ===")
    for m in address_only_examples[:N_EXAMPLES]:
        print(f"{m['query'][:45]} | {m['answer']} | k_rank {m['k_rank']} k_top1 {m['k_top1']}")

    print(f"\n=== korean_only 사례 (상위 {N_EXAMPLES}건) ===")
    for m in korean_only_examples[:N_EXAMPLES]:
        print(f"{m['query'][:45]} | {m['answer']} | a_rank {m['a_rank']} a_top1 {m['a_top1']}")

    print(f"\n=== both_wrong 사례, 두 모델 중 더 나은 순위 기준 오름차순 (상위 {N_EXAMPLES}건) ===")
    for m in both_wrong_examples[:N_EXAMPLES]:
        print(f"{m['query'][:45]} | {m['answer']} | "
              f"k_rank {m['k_rank']} k_top1 {m['k_top1'][:35]} | "
              f"a_rank {m['a_rank']} a_top1 {m['a_top1'][:35]}")


if __name__ == "__main__":
    main()
