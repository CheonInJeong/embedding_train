from pathlib import Path

# src/config.py 기준 저장소 루트 (실행 위치(cwd)에 상관없이 항상 같은 경로를 가리키게 하기 위함)
ROOT_DIR = Path(__file__).resolve().parent.parent

# MODEL_NAME = "BAAI/bge-m3"
MODEL_NAME = "upskyy/bge-m3-korean"
MODEL_PATH = str(ROOT_DIR / "models" / "bge-m3-korean")

TRAIN_FAIL = str(ROOT_DIR / "data" / "train" / "fail.txt")
TRAIN_ANSWER = str(ROOT_DIR / "data" / "train" / "answer.txt")

VALID_FAIL = str(ROOT_DIR / "data" / "valid" / "fail.txt")
VALID_ANSWER = str(ROOT_DIR / "data" / "valid" / "answer.txt")

TEST_FAIL = str(ROOT_DIR / "data" / "test" / "fail.txt")
TEST_ANSWER = str(ROOT_DIR / "data" / "test" / "answer.txt")

OUTPUT_DIR = str(ROOT_DIR / "models" / "bge-m3-address")

BATCH_SIZE = 32

EPOCH = 3

LEARNING_RATE = 2e-5
