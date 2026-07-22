import os

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments

from dataset import load_dataset, load_triplet_dataset
import config

MODEL_NAME = config.MODEL_NAME

OUTPUT_DIR = config.OUTPUT_DIR

EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 2e-5


def main():

    print("Loading model...")
    if os.path.isdir(config.MODEL_PATH):
        model = SentenceTransformer(config.MODEL_PATH)
    else:
        # GPU 서버 등 로컬에 base model 캐시가 없는 환경에서는 HF Hub에서 받는다.
        print(f"{config.MODEL_PATH} 없음 - HuggingFace Hub({config.MODEL_NAME})에서 다운로드")
        model = SentenceTransformer(config.MODEL_NAME)

    print("Loading dataset...")
    # 기존 증강 데이터 (anchor, positive)
    pair_dataset = load_dataset(config.TRAIN_FAIL, config.TRAIN_ANSWER)
    # 번지수 근접 오답 hard negative (anchor, positive, negative)
    hard_neg_dataset = load_triplet_dataset(config.TRAIN_HARD_NEG)
    valid_dataset = load_dataset(config.VALID_FAIL, config.VALID_ANSWER)

    print(pair_dataset)
    print(hard_neg_dataset)
    print(valid_dataset)

    # pair 데이터셋과 triplet 데이터셋은 컬럼 수가 달라 하나로 합칠 수 없으므로,
    # 서브셋 이름을 key로 하는 dict로 넘긴다. 같은 MultipleNegativesRankingLoss
    # 인스턴스가 negative 컬럼이 있으면 명시적 negative까지, 없으면 배치 내
    # 다른 positive만 negative로 사용한다.
    train_dataset = {
        "pairs": pair_dataset,
        "hard_negatives": hard_neg_dataset,
    }

    loss = MultipleNegativesRankingLoss(model) #다른 positive들을 모두 negative로 취급
    loss = {
        "pairs": loss,
        "hard_negatives": loss,
    }
    print(loss)

    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,

        num_train_epochs=EPOCHS,

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,

        learning_rate=LEARNING_RATE,

        warmup_ratio=0.1,

        fp16=False,
        bf16=True,

        eval_strategy="epoch",
        save_strategy="epoch",

        save_total_limit=3,

        load_best_model_at_end=True,

        logging_steps=1,

        logging_dir=f"{OUTPUT_DIR}/logs",

        report_to="none",
    )

    print(args)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset={"pairs": valid_dataset},
        loss=loss,
    )

    print("Start Training")
    trainer.train()

    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)

    print("Done")


if __name__ == "__main__":
    main()