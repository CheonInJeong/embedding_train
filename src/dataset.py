from datasets import Dataset

def load_dataset(fail_file, answer_file):
    with open(fail_file, encoding="utf-8") as f:
        fail = [
            line.strip()
            for line in f
            if line.strip()
        ]
    with open (answer_file, encoding="utf-8") as f:
        answer = [
            line.strip()
            for line in f
            if line.strip()
        ]

    assert len(fail) == len(answer)

    anchors = []
    positives = []
    for f, a in zip(fail, answer):
        anchors.append(f)
        positives.append(a)

    return Dataset.from_dict({
        "anchor": anchors,
        "positive": positives,
    })