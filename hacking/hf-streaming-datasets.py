from datasets import load_dataset
from torch.utils.data import DataLoader

ds = load_dataset(
    "MLCommons/peoples_speech", split="train", streaming=True, trust_remote_code=True
).with_format("torch")

batch_size = 64
num_workers = 4

collate = lambda x: x

dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, collate_fn=collate)


import itertools
import time

n = 100
t1 = time.perf_counter()
for batch in itertools.islice(dl, n):
    print(batch)

t2 = time.perf_counter()

elapsed = t2 - t1
samples = batch_size * n
throughput = samples / elapsed

print(f"{elapsed=:.2f} {throughput=:.1f}samples/s")
