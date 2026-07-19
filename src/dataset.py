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