from datasets import load_dataset
from itertools import islice
import matplotlib.pyplot as plt
from tqdm import tqdm

train_ds = load_dataset(
    "mozilla-foundation/common_voice_16_1",
    "en",
    split="train",
    trust_remote_code=True,
)

n = 100_000

with open("sentences.txt", "w") as f:
    for item in tqdm(islice(train_ds, n), total=n):
        f.write(item["sentence"] + "\n")
