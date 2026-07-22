import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 같은 도로명(prefix)에서 번지수만 가까운(1~MAX_DIFF 차이) "근접 오답" 주소를 찾아
# (발화, 정답주소, 오답주소) triplet을 만든다.
#
# 배경: DB 임베딩 검색 평가에서 파인튜닝 모델이 오답을 내는 경우 대부분
# "도로명은 맞는데 번지수만 다른 후보"였다. MultipleNegativesRankingLoss는
# 배치 내 다른 positive들만 자동으로 negative로 쓰기 때문에, 같은 배치에
# 우연히 같은 도로명 주소가 없으면 이런 근접 오답을 구분하는 신호를 전혀
# 받지 못한다. 여기서 만든 explicit negative 컬럼을 데이터셋에 추가하면
# 이 케이스를 직접적으로 학습시킬 수 있다.

GALLERY_FILE = "data/address/임베딩데이터(읍면리포함).txt"

ADDR_PATTERN = re.compile(r"^(.*?)(\d+)(-\d+)?$")

MAX_DIFF = 5  # 번지수 차이가 이 값 이하인 것만 "근접 오답"으로 인정


def parse_address(addr):
    m = ADDR_PATTERN.match(addr)
    if not m:
        return None, None
    prefix, number, _sub = m.groups()
    return prefix, int(number)


def build_gallery_index():
    index = defaultdict(list)
    with open(GALLERY_FILE, encoding="utf-8") as f:
        for line in f:
            addr = line.strip()
            if not addr:
                continue
            prefix, number = parse_address(addr)
            if prefix is None:
                continue
            index[prefix].append((number, addr))
    return index


def find_hard_negative(index, positive_addr):
    prefix, number = parse_address(positive_addr)
    if prefix is None:
        return None

    candidates = index.get(prefix, [])
    if len(candidates) <= 1:
        return None

    best = None
    best_diff = None
    for cand_number, cand_addr in candidates:
        if cand_addr == positive_addr:
            continue
        diff = abs(cand_number - number)
        if diff == 0:
            continue
        if diff > MAX_DIFF:
            continue
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best = cand_addr

    return best


def build_triplets(anchor_file, output_file):
    index = build_gallery_index()
    print(f"gallery index: {len(index)}개 도로명 prefix")

    total = 0
    with_negative = 0

    with open(anchor_file, encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            query, answer = line.split("\t")
            total += 1

            negative = find_hard_negative(index, answer)
            if negative is None:
                continue

            with_negative += 1
            f_out.write(f"{query}\t{answer}\t{negative}\n")

    print(f"총 {total}건 중 {with_negative}건에서 근접 오답(hard negative) 발견")
    print(f"저장 위치: {output_file}")


if __name__ == "__main__":
    # 사용법: python -m scripts.build_hard_negatives [입력_anchor_파일] [출력_파일]
    # 입력 파일 형식은 "발화\t정답주소" 탭 구분 (기본값: STT발화_정답_학습데이터.txt)
    default_anchor = "data/address/STT발화_정답_학습데이터.txt"
    default_output = "data/address/STT발화_정답_hard_neg.txt"

    anchor_file = sys.argv[1] if len(sys.argv) > 1 else default_anchor
    output_file = sys.argv[2] if len(sys.argv) > 2 else default_output

    build_triplets(anchor_file, output_file)
