from pathlib import Path
import argparse
import random

# ==========================
# 설정
# ==========================

RANDOM_SEED = 42
TRAIN_RATIO = 0.8

FAIL_FILE = Path("C:\\Users\\injeong\\PycharmProjects\\finetuning\\bge-finetuning\\data\\fail.txt")
ANSWER_FILE = Path("C:\\Users\\injeong\\PycharmProjects\\finetuning\\bge-finetuning\\data\\answer.txt")

TRAIN_DIR = Path("C:\\Users\\injeong\\PycharmProjects\\finetuning\\bge-finetuning\\data\\train")
VALID_DIR = Path("C:\\Users\\injeong\\PycharmProjects\\finetuning\\bge-finetuning\\data\\valid")


def load_lines(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def save_lines(path: Path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def filter_region(data, region):
    """
    region:
        ALL
        부산
        서울
        대구
        ...
    """

    if region.upper() == "ALL":
        return data

    region = region.strip()

    return [
        (fail, answer)
        for fail, answer in data
        if region in answer
    ]

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--region",
        default="ALL",
        help="예: 부산, 서울특별시, ALL"
    )

    args = parser.parse_args()

    fail = load_lines(FAIL_FILE)
    answer = load_lines(ANSWER_FILE)

    if len(fail) != len(answer):
        raise ValueError("fail.txt와 answer.txt의 라인 수가 다릅니다.")

    data = list(zip(fail, answer))

    # 지역 필터
    data = filter_region(data, args.region)

    if len(data) == 0:
        print(f"[ERROR] '{args.region}' 데이터가 없습니다.")
        return

    random.seed(RANDOM_SEED)
    random.shuffle(data)

    split = int(len(data) * TRAIN_RATIO)

    train = data[:split]
    valid = data[split:]

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VALID_DIR.mkdir(parents=True, exist_ok=True)

    save_lines(TRAIN_DIR / "fail.txt", [x[0] for x in train])
    save_lines(TRAIN_DIR / "answer.txt", [x[1] for x in train])

    save_lines(VALID_DIR / "fail.txt", [x[0] for x in valid])
    save_lines(VALID_DIR / "answer.txt", [x[1] for x in valid])

    print("======================================")
    print(f"Region      : {args.region}")
    print(f"Total       : {len(data)}")
    print(f"Train       : {len(train)}")
    print(f"Validation  : {len(valid)}")
    print("======================================")


if __name__ == "__main__":
    main()