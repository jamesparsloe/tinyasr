from datasets import load_dataset
from itertools import islice
import matplotlib.pyplot as plt
import pandas as pd


train_ds = load_dataset(
    "mozilla-foundation/common_voice_16_1",
    "en",
    split="train",
    trust_remote_code=True,
)

sample_size = 10_000

rows = []

for item in islice(train_ds, sample_size):
    duration = item["audio"]["array"].shape[-1] / item["audio"]["sampling_rate"]
    text_len = len(item["sentence"])

    rows.append({"duration": duration, "text_len": text_len})

df = pd.DataFrame(rows)

# print summary
print(df.describe())

# plot histogram of text lengths
fig, ax = plt.subplots()
ax.hist(df["text_len"], bins=100)
ax.set_xlabel("text len")
fig.savefig("text_len-dist.png")


# plot histogram of audio durations
fig, ax = plt.subplots()
ax.hist(df["duration"], bins=100)
ax.set_xlabel("duration")
fig.savefig("duration-dist.png")
