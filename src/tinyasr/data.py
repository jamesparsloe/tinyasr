import functools
import re

import torch
from datasets import Audio, interleave_datasets, load_dataset
from torchaudio.functional import resample
from torchdata.datapipes.iter import IterableWrapper, SampleMultiplexer

CONFIGS = {
    "train": [
        # {
        #     "path": "MLCommons/peoples_speech",
        #     "split": "train",
        #     "weight": 30.0,
        # },
        {
            "path": "mozilla-foundation/common_voice_16_1",
            "name": "en",
            "split": "train",
            "weight": 1.0,
        },
        {
            "path": "librispeech_asr",
            "split": "train.clean.100",
            "weight": 0.1,
        },
        {
            "path": "librispeech_asr",
            "split": "train.clean.360",
            "weight": 0.36,
        },
        {
            "path": "librispeech_asr",
            "split": "train.other.500",
            "weight": 0.5,
        },
        {
            "path": "facebook/voxpopuli",
            "name": "en",
            "split": "train",
            "weight": 0.5,
        },
    ],
    "val": [
        # {
        #     "path": "MLCommons/peoples_speech",
        #     "split": "validation",
        #     "weight": 1.0,
        # },
        {
            "path": "mozilla-foundation/common_voice_16_1",
            "name": "en",
            "split": "validation",
            "weight": 1.0,
        },
        {
            "path": "librispeech_asr",
            "split": "validation.clean",
            "weight": 1.0,
        },
        {
            "path": "librispeech_asr",
            "split": "validation.other",
            "weight": 1.0,
        },
        {
            "path": "facebook/voxpopuli",
            "name": "en",
            "split": "validation",
            "weight": 1.0,
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


def transform(sample, *, new_sample_rate: int):
    keys = ["text", "duration", "waveform", "sample_rate"]

    if sample["sample_rate"] != new_sample_rate:
        waveform = resample(sample["waveform"], sample["sample_rate"], new_sample_rate)
        sample["waveform"] = waveform
        sample["sample_rate"] = new_sample_rate

    return {k: sample[k] for k in keys}


def duration_filter(sample, max_duration: float = 15.0):
    return sample["duration"] <= max_duration


def build_dp(
    split: str = "train",
    sample_rate: int = 16_000,
    max_duration: float = 15.0,
    seed: int = 42,
):
    datasets = {}

    configs = CONFIGS[split]

    for config in configs:
        weight = config["weight"]
        del config["weight"]
        dp = load_dataset(**config, streaming=True, trust_remote_code=True).map(
            hf_transform
        )
        dp = IterableWrapper(dp).prefetch(64)
        datasets[dp] = weight

    dp = SampleMultiplexer(datasets, seed=seed)

    # eject from HF and wrap in a torchdata iterable-style dp
    _filter = functools.partial(duration_filter, max_duration=max_duration)
    _transform = functools.partial(transform, new_sample_rate=sample_rate)
    dp = dp.map(_transform).filter(_filter)

    return dp


if __name__ == "__main__":
    from itertools import islice

    ds = build_dp()
    print(ds)
    for item in islice(ds, 1000):
        print(item)
