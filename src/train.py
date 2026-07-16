from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments

from dataset import load_dataset
import config

MODEL_NAME = config.MODEL_NAME


OUTPUT_DIR = "../output/bge-m3-address"

EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 2e-5


def main():

    print("Loading model...")
    model = SentenceTransformer(config.MODEL_PATH)

    print("Loading dataset...")
    train_dataset = load_dataset(config.TRAIN_FAIL, config.TRAIN_ANSWER )
    valid_dataset = load_dataset(config.VALID_FAIL, config.VALID_ANSWER )

    print(train_dataset)
    print(valid_dataset)

    loss = MultipleNegativesRankingLoss(model) #다른 positive들을 모두 negative로 취급
    print(loss)

    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,

        num_train_epochs=EPOCHS,

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,

        learning_rate=LEARNING_RATE,

        warmup_ratio=0.1,

        fp16=False,
        bf16=False,

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
        eval_dataset=valid_dataset,
        loss=loss,
    )

    print("Start Training")
    trainer.train()

    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)

    print("Done")


if __name__ == "__main__":
    main()