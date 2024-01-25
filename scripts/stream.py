from datasets import load_dataset

train_ds = load_dataset(
    "mc4", "en", split="train", streaming=True, trust_remote_code=True
)

print(next(iter(train_ds)))
