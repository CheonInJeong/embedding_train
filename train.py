from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
import sentence_transformers.sentence_transformer.losses as losses

from src import config
from src.dataset import load_dataset

model = SentenceTransformer(config.MODEL_NAME)

train_dataset = load_dataset(config.TRAIN_FAIL, config.TRAIN_ANSWER)

loss = losses.MultipleNegativesRankingLoss(model)

args = SentenceTransformerTrainingArguments(
    output_dir=config.OUTPUT_DIR,
    num_train_epochs=config.EPOCH,
    per_device_train_batch_size=config.BATCH_SIZE,
    warmup_steps=100,
    learning_rate=config.LEARNING_RATE,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
)

trainer.train()

model.save(config.OUTPUT_DIR)
