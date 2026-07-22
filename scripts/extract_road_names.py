import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 주소 텍스트를 한 줄씩 읽어 공백으로 나눈 토큰 중 '로'/'길'/'번길'로 끝나는
# 단어(도로명)를 추출한다.
#
# "1순환로1137번길 20"처럼 도로명이 한 토큰에 붙어있으면 그 토큰이 곧 도로명이지만,
# "선릉로 100길 5"처럼 '-로'와 '숫자+길'이 공백으로 분리된 경우도 있어서
# '로'로 끝나는 토큰을 만나면 바로 다음 토큰이 "숫자(번)?길" 패턴인지 확인하고,
# 맞으면 두 토큰을 합쳐 하나의 도로명으로 취급한다.
NUM_GIL_PATTERN = re.compile(r"^\d+(번)?길$")


def extract_road_names_from_line(tokens):
    names = []
    i = 0
    n = len(tokens)

    while i < n:
        token = tokens[i]

        if token.endswith("로"):
            if i + 1 < n and NUM_GIL_PATTERN.match(tokens[i + 1]):
                names.append(token + tokens[i + 1])
                i += 2
            else:
                names.append(token)
                i += 1
        elif token.endswith("길"):
            names.append(token)
            i += 1
        else:
            i += 1

    return names


def extract_road_names(input_file):
    road_names = {}  # dict로 순서 보존 + 중복 제거

    with open(input_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            for name in extract_road_names_from_line(line.split()):
                road_names[name] = None

    return list(road_names.keys())


def main(input_file, output_file):
    road_names = extract_road_names(input_file)

    with open(output_file, "w", encoding="utf-8") as f:
        for name in road_names:
            f.write(name + "\n")

    print(f"총 {len(road_names)}개 도로명 추출")
    print(f"저장 위치: {output_file}")


if __name__ == "__main__":
    # 사용법: python -m scripts.extract_road_names [입력_주소_파일] [출력_파일]
    default_input = "../data/주소_20260722.txt"
    default_output = "../data/주소_20260722_도로명.txt"

    input_file = sys.argv[1] if len(sys.argv) > 1 else default_input
    output_file = sys.argv[2] if len(sys.argv) > 2 else default_output

    main(input_file, output_file)
