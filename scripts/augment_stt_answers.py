import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agument import augment

INPUT_FILE = Path("data/address/STT발화_정답_학습데이터.txt")
OUTPUT_FILE = Path("data/address/STT발화_정답_anchor.txt")
SAMPLES_PER_ADDRESS = 25


def load_answers():
    addresses = []
    seen = set()
    with open(INPUT_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            _, address = line.split("\t")
            if address not in seen:
                seen.add(address)
                addresses.append(address)
    return addresses


def main():
    addresses = load_answers()
    total = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for address in addresses:
            samples = augment(address, SAMPLES_PER_ADDRESS)
            samples = list(dict.fromkeys(samples))

            for sample in samples:
                out.write(f"{sample}\t{address}\n")
                total += 1

    print("====================================")
    print(f"원본 주소 수 : {len(addresses)}")
    print(f"생성된 데이터 : {total}")
    print(f"저장 위치 : {OUTPUT_FILE}")
    print("====================================")


if __name__ == "__main__":
    main()
