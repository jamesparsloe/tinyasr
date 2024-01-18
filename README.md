# tinyasr

Hacking on tiny decoder-only transformers for ASR. This is _very much_ a toy project - I don't have much compute at the moment (just a single 2080 Ti 😢), and only have access to the standard and easily attainable ASR datasets (LibriSpeech, Common Voice etc). I'd just like to scratch an itch to see how close to an equivalently sized Whisper model I can get with something like this (and hopefully learn some things along the way too).

## Getting Started

```sh
python -m venv env
source env/bin/activate
python -m pip install -e .
```

## Training

```sh
python -m tinyasr.train configs/base.yaml
```

## Experiments

I'll be logging my experiments [here](<https://wandb.ai/jamesparsloe/tinyasr>).
