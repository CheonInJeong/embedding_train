import torch
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)

from src import config
from src.dataset import load_anchor_dataset

# Apple M1/M2/M3 MPS GPU 우선, 없으면 CUDA, 없으면 CPU
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")

model = SentenceTransformer(config.MODEL_NAME, device=device)

train_dataset = load_anchor_dataset(config.ANCHOR_FILE)

print(f"Train samples: {len(train_dataset)}")

loss = losses.MultipleNegativesRankingLoss(model)

args = SentenceTransformerTrainingArguments(
    output_dir=config.OUTPUT_DIR,
    num_train_epochs=config.EPOCH,
    per_device_train_batch_size=config.BATCH_SIZE,
    warmup_steps=100,
    learning_rate=config.LEARNING_RATE,
    fp16=False,   # MPS는 fp16 미지원
    bf16=False,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
)

trainer.train()

model.save(config.OUTPUT_DIR)
print(f"Model saved to {config.OUTPUT_DIR}")
