from datasets import Dataset


def load_dataset(fail_file, answer_file):
    with open(fail_file, encoding="utf-8") as f:
        fail = [line.strip() for line in f if line.strip()]
    with open(answer_file, encoding="utf-8") as f:
        answer = [line.strip() for line in f if line.strip()]

    assert len(fail) == len(answer)

    return Dataset.from_dict({
        "anchor": fail,
        "positive": answer,
    })


def load_triplet_dataset(triplet_file):
    """
    탭(\t)으로 구분된 (STT발화, 정답주소, 오답주소) triplet을 읽어 Dataset 반환.
    형식: STT발화\t정답주소\t오답주소

    오답주소는 정답과 도로명이 같고 번지수만 가까운 hard negative
    (scripts/build_hard_negatives.py 참고). MultipleNegativesRankingLoss는
    "negative" 컬럼이 있으면 배치 내 다른 positive들에 더해 이 명시적
    negative까지 함께 사용하므로, 번지수 근접 오답 구분력을 직접 학습시킬 수 있다.
    """
    anchors = []
    positives = []
    negatives = []

    with open(triplet_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            anchors.append(parts[0])
            positives.append(parts[1])
            negatives.append(parts[2])

    return Dataset.from_dict({
        "anchor": anchors,
        "positive": positives,
        "negative": negatives,
    })


def load_anchor_dataset(anchor_file):
    """
    탭(\t)으로 구분된 (STT발화, 정답주소) 쌍을 읽어 Dataset 반환
    형식: STT발화\t정답주소
    """
    anchors = []
    positives = []

    with open(anchor_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", maxsplit=1)
            if len(parts) != 2:
                continue
            anchors.append(parts[0])
            positives.append(parts[1])

    return Dataset.from_dict({
        "anchor": anchors,
        "positive": positives,
    })