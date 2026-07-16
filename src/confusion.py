import random

CONFUSION = {
    "센텀": [
        ("센터", 0.60),
        ("샌텀", 0.20),
    ],
    "테헤란": [
        ("태해란", 0.30),
        ("테란", 0.20),
    ],
    "해운대": [
        ("해운데", 0.15),
    ],
    "번길": [
        ("번 기리", 0.10),
    ],
    "로": [
        ("노", 0.02),
    ],
}


def apply_confusion(text: str) -> str:
    """
    STT 오인식 패턴을 확률적으로 적용한다.
    한 단어당 최대 한 번만 치환한다.
    """
    result = text

    for original, candidates in CONFUSION.items():
        if original not in result:
            continue

        r = random.random()
        cumulative = 0.0

        for replacement, prob in candidates:
            cumulative += prob
            if r < cumulative:
                result = result.replace(original, replacement, 1)
                break

    return result