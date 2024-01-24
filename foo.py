from datasets import load_dataset

train_ds = load_dataset("mc4", "en", split="train", trust_remote_code=True)
# val_ds = load_dataset("mc4", "en", split="validation", trust_remote_code=True)

print(next(iter(train_ds)))
