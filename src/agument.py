# 학습 데이터 생성
# 1. 광역시 축약
# 2. 구 제거
# 3. 숫자 한글화
# 4. 번지 표현
# 5. 띄어쓰기 제거
# 6. prefix 여보세요, 저기, 여기, 주소는 등
# 7. suffix 입니다, 인데요, 요, 이에요 등
# 8. 조사
# 9. 자연스러운 말
# 10. 불필요한 단어 삽입
# 11. 숫자혼합
# 12. 띄어쓰기 흔들기
# 13. 행정구역 제거
# 14. 공백 제거
# 15. stt 특유 오류


## 자주 발생하는 치환 패턴(예: "센텀→센터", "광역시 생략", 숫자 읽기 등)을 자동으로 추출하는 스크립트를 만들기
import random
import re
from pathlib import Path

from src.confusion import apply_confusion

# -----------------------------
# Prefix / Suffix
# -----------------------------

PREFIXES = [
    "",
    "여기 ",
    "주소는 ",
    "주소가 ",
    "제가 있는 곳은 ",
    "지금 ",
    "혹시 ",
    "저기 ",
    "네 ",
    "예 ",
]

SUFFIXES = [
    "",
    " 입니다",
    " 인데요",
    " 이에요",
    " 맞아요",
    " 거든요",
    " 요",
]

# -----------------------------
# 광역시 축약
# -----------------------------

CITY_RULE = {
    "서울특별시": "서울",
    "부산광역시": "부산",
    "대구광역시": "대구",
    "인천광역시": "인천",
    "광주광역시": "광주",
    "대전광역시": "대전",
    "울산광역시": "울산",
    "세종특별자치시": "세종",
    "전북특별자치도" : "전북",
    "경상북도":"경북",
    "전남광주통합특별시":"전남",
    "경상남도":"경남",
    "경기도":"경기"
}


# -----------------------------
# 숫자 -> 한글
# (9999 정도까지)
# -----------------------------

NUM = {
    0: "",
    1: "일",
    2: "이",
    3: "삼",
    4: "사",
    5: "오",
    6: "육",
    7: "칠",
    8: "팔",
    9: "구",
}

UNIT = [
    "",
    "십",
    "백",
    "천"
]

BIG_UNIT = [
    "",
    "만",
    "억",
    "조"
]


def _four_digit_to_korean(num: int):

    if num == 0:
        return ""

    result = ""

    s = str(num)

    length = len(s)

    for idx, ch in enumerate(s):

        n = int(ch)

        if n == 0:
            continue

        unit = UNIT[length - idx - 1]

        if n == 1 and unit != "":
            result += unit
        else:
            result += NUM[n] + unit

    return result


def number_to_korean(num: int):
    # 도로명 번호에 동/호수가 공백만 사이에 두고 붙어있으면(remove_space규칙 등으로) 4자리를 넘는 숫자가 합쳐질 수 있어, 4자리씩 끊어 만/억 단위로 처리한다. (기존에는 UNIT이 4자리까지만 지원해서 5자리 이상 숫자가 들어오면 IndexError가 났다)

    if num == 0:
        return "영"

    groups = []

    n = num

    while n > 0:
        groups.append(n % 10000)
        n //= 10000

    result = ""

    for i in range(len(groups) - 1, -1, -1):

        g = groups[i]

        if g == 0:
            continue

        result += _four_digit_to_korean(g) + BIG_UNIT[i]

    return result


# -----------------------------
# Rule
# -----------------------------

def shorten_city(addr):

    result = [addr]

    for k, v in CITY_RULE.items():

        if k in addr:
            result.append(addr.replace(k, v))

    return result


# CITY_RULE 값 중 자치구 접미사와 형태가 겹치는 광역시 축약형
# (예: "대구"가 "대구광역시"의 일부가 아니라 단독 토큰으로 남아있을 때
#  자치구 "구"로 오인되어 "대"로 잘리는 것을 방지)
_GU_SUFFIX_EXCEPTIONS = {"대구"}


def remove_gu(addr):

    def _strip(m):
        word = m.group(0)

        if word in _GU_SUFFIX_EXCEPTIONS:
            return word

        return m.group(1)

    # 앞뒤로 한글이 이어지지 않는(=토큰 전체가 "구"로 끝나는) 경우만 매치.
    # 이렇게 해야 "대구광역시", "대구노원한신더휴"처럼 "구"가 단어 중간에
    # 우연히 낀 경우까지 잘못 잘라내는 것을 막을 수 있다.
    return [
        addr,
        re.sub(r"(?<![가-힣])([가-힣]+)구(?![가-힣])", _strip, addr)
    ]


def remove_space(addr):

    return [
        addr,
        re.sub(r"\s+", "", addr),
        re.sub(r"(\D)\s+(\d)", r"\1\2", addr)
    ]


def number_variation(addr):

    results = [addr]

    nums = re.findall(r"\d+", addr)

    for n in nums:

        # 앞뒤로 다른 숫자가 붙어있지 않은, 정확히 이 번호(n)만 치환한다.
        # addr.replace(n, ...)를 그대로 쓰면 "12", "16" 안에 있는 "1"까지
        # 걸려서 서로 다른 번호가 뒤섞여버리는 문제가 있었다.
        pattern = re.compile(r"(?<!\d)" + re.escape(n) + r"(?!\d)")

        korean = number_to_korean(int(n))

        results.append(
            pattern.sub(korean, addr)
        )

        for suffix in ("번지", "번", "호"):

            # "1502호"처럼 번호 뒤에 이미 같은 접미사가 있으면 또 붙이지
            # 않는다. 그대로 두면 "1502호호"처럼 글자가 중복된다.
            if re.search(r"(?<!\d)" + re.escape(n) + re.escape(suffix), addr):
                continue

            results.append(
                pattern.sub(n + suffix, addr)
            )

    return results


def remove_region(addr):

    result = [addr]

    result.append(
        re.sub(r"^[가-힣]+(특별시|광역시)\s*", "", addr)
    )

    return result


# -----------------------------
# 최종 증강
# -----------------------------

RULES = [
    shorten_city,
    remove_gu,
    remove_space,
    number_variation,
    remove_region,
]


def augment(address, max_samples=30):

    candidates = {address}

    # Rule 1개 적용
    for rule in RULES:

        for text in rule(address):
            candidates.add(text)

    # Rule 2개 조합
    for rule1 in RULES:

        for t in rule1(address):

            for rule2 in RULES:

                # 같은 규칙을 자기 출력에 다시 적용하지 않는다.
                # number_variation처럼 원본 숫자를 남겨두는 규칙은
                # 재적용 시 "번지번지", "호호"처럼 접미사가 중첩된다.
                if rule2 is rule1:
                    continue

                for t2 in rule2(t):

                    candidates.add(t2)

    final = set()

    for c in candidates:
        c = apply_confusion(c)
        prefix = random.choice(PREFIXES)
        suffix = random.choice(SUFFIXES)

        final.add(prefix + c + suffix)

    final = list(final)

    random.shuffle(final)

    return final[:max_samples]

# -----------------------------
# 테스트
# -----------------------------

if __name__ == "__main__":
    INPUT_FILE = Path("../data/address/STT주소.txt")
    OUTPUT_FILE = Path("../data/address/STT주소_anchor.txt")
    SAMPLES_PER_ADDRESS = 20

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        addresses = [line.strip() for line in f if line.strip()]

    total = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

        for address in addresses:

            samples = augment(address, SAMPLES_PER_ADDRESS)

            # 중복 제거
            samples = list(dict.fromkeys(samples))

            for sample in samples:
                out.write(f"{sample}\t{address}\n")
                print(sample)
                total += 1

    print("====================================")
    print(f"원본 주소 수 : {len(addresses)}")
    print(f"생성된 데이터 : {total}")
    print(f"저장 위치 : {OUTPUT_FILE}")
    print("====================================")