import functools
import re

import torch
from datasets import Audio, interleave_datasets, load_dataset
from torch.utils.data import ConcatDataset, Dataset
from torchaudio.functional import resample
from torchdata.datapipes.iter import IterableWrapper, SampleMultiplexer
from torchdata.datapipes.map import SequenceWrapper

# TODO skipping People's Speech for now - it's too big for my machine and I can't get the
# streaming datasets to play nicely as I'd hoped
CONFIGS = {
    "train": [
        # {
        #     "path": "MLCommons/peoples_speech",
        #     "split": "train",
        # },
        {
            "path": "mozilla-foundation/common_voice_16_1",
            "name": "en",
            "split": "train",
        },
        {
            "path": "librispeech_asr",
            "split": "train.clean.100",
        },
        {
            "path": "librispeech_asr",
            "split": "train.clean.360",
        },
        {
            "path": "librispeech_asr",
            "split": "train.other.500",
        },
        {
            "path": "facebook/voxpopuli",
            "name": "en",
            "split": "train",
        },
    ],
    "val": [
        # {
        #     "path": "MLCommons/peoples_speech",
        #     "split": "validation",
        # },
        {
            "path": "mozilla-foundation/common_voice_16_1",
            "name": "en",
            "split": "validation",
        },
        {
            "path": "librispeech_asr",
            "split": "validation.clean",
        },
        {
            "path": "librispeech_asr",
            "split": "validation.other",
        },
        {
            "path": "facebook/voxpopuli",
            "name": "en",
            "split": "validation",
        },
    ],
}


def hf_transform(sample):
    waveform = torch.from_numpy(sample["audio"]["array"]).to(torch.float32)
    sample_rate = sample["audio"]["sampling_rate"]

    duration = waveform.size(-1) / sample_rate

    text = (
        sample.get("raw_text")
        or sample.get("text")
        or sample.get("sentence")
        or sample.get("normalized_text")
    )
    assert text is not None, f"Missing text for {sample}"

    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)

    return {
        "text": text,
        "duration": duration,
        "waveform": waveform,
        "sample_rate": sample_rate,
    }


def transform(sample, *, new_sample_rate: int, max_duration: float):
    waveform = torch.from_numpy(sample["audio"]["array"]).to(torch.float32)
    sample_rate = sample["audio"]["sampling_rate"]

    duration = waveform.size(-1) / sample_rate

    if duration > max_duration:
        return None

    text = (
        sample.get("raw_text")
        or sample.get("text")
        or sample.get("sentence")
        or sample.get("normalized_text")
    )
    assert text is not None, f"Missing text for {sample}"

    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)

    if sample_rate != new_sample_rate:
        waveform = resample(waveform, sample_rate, new_sample_rate)

    return {
        "duration": duration,
        "text": text,
        "waveform": waveform,
        "sample_rate": new_sample_rate,
    }


def filter_none(sample):
    return sample is not None


def to_iter_datapipe(ds: Dataset):
    dp = SequenceWrapper(ds).to_iter_datapipe()
    return dp


def build_dp(
    split: str = "train",
    sample_rate: int = 16_000,
    max_duration: float = 15.0,
    seed: int = 42,
):
    configs = CONFIGS[split]

    datasets = {}

    n = 0
    for config in configs:
        path = config["path"]
        split = config["split"]
        ds = load_dataset(**config, trust_remote_code=True)
        weight = len(ds)
        print(f"Loading {path} {split} {weight=}")
        n += weight
        dp = to_iter_datapipe(ds)
        datasets[dp] = weight

    print(f"{n} items in dataset")

    dp = SampleMultiplexer(datasets, seed=seed)

    # eject from HF and wrap in a torchdata iterable-style dp
    _transform = functools.partial(
        transform, new_sample_rate=sample_rate, max_duration=max_duration
    )
    dp = dp.map(_transform).filter(filter_none)

    return dp


def main():
    for split in ["train", "val"]:
        dp = build_dp(split=split)


if __name__ == "__main__":
    main()
