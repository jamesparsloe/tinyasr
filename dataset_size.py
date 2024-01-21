from datasets import load_dataset, interleave_datasets
from itertools import islice
import matplotlib.pyplot as plt
from tqdm import tqdm

ds = []

for lang in ["en", "es", "fr", "de", "it"]:
    d = load_dataset(
        "mozilla-foundation/common_voice_16_1",
        lang,
        split="train",
        trust_remote_code=True,
    )
    print(f"{lang=} {len(d)} samples")
    ds.append(d)

train_ds = interleave_datasets(ds, stopping_strategy="all_exhausted")

batch_size = 64
samples = len(train_ds)
batches_per_epoch = samples // batch_size

print(f"{samples=} {batch_size=} {batches_per_epoch=}")
