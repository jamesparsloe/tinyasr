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

batch_size = 64
samples = len(train_ds)
batches_per_epoch = samples // batch_size

print(f"{samples=} {batch_size=} {batches_per_epoch=}")
