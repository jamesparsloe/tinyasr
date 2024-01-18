# tinyasr

A toy decoder-only version of Whisper just to get a bit more intuition around transformer-based ASR models.

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
