from pathlib import Path

import config

# agument.py가 만든 "발화\t정답주소" 형식의 anchor 파일을
# split_dataset.py / train.py가 기대하는 fail.txt / answer.txt
# (줄 번호로 정렬된 두 개의 파일) 형식으로 변환한다.

ANCHOR_FILE = config.ROOT_DIR / "data" / "address" / "STT주소_anchor.txt"
FAIL_FILE = config.ROOT_DIR / "data" / "fail.txt"
ANSWER_FILE = config.ROOT_DIR / "data" / "answer.txt"


def main():
    fail = []
    answer = []

    with open(ANCHOR_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if not line.strip():
                continue

            sample, address = line.split("\t")

            fail.append(sample)
            answer.append(address)

    with open(FAIL_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(fail))

    with open(ANSWER_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(answer))

    print("======================================")
    print(f"원본 : {ANCHOR_FILE}")
    print(f"변환된 줄 수 : {len(fail)}")
    print(f"저장 위치 : {FAIL_FILE}, {ANSWER_FILE}")
    print("======================================")


if __name__ == "__main__":
    main()
