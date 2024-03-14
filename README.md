# tinyasr

Hacking on tiny decoder-only transformers for ASR. I'm compute limited by what I can _reasonably_ train on my 2 4090s and/or Colab A100s. If anything I'm probably more constrained on the data side of things - I only have access to the standard and easily attainable ASR datasets (LibriSpeech, Common Voice etc) and not the magical 680k hour data that Whisper was trained with.

I'd just like to scratch an itch to see how close to an equivalently sized Whisper model I can get with something like this (and hopefully learn some things along the way too).

## Getting Started

```sh
python -m venv env
source env/bin/activate
python -m pip install -e .[dev]
```

## Training

```sh
python -m tinyasr.data # download the datasets - takes a while, probably best to do it in a tmux session
python -m tinyasr.train configs/base.yaml
```

## Experiments

I'll be logging my experiments [here](<https://wandb.ai/jamesparsloe/tinyasr>).

## Roadmap

- [x] Make it work
- [ ] Proper benchmarks
- [ ] Text tokenization - just using a utf-8 based tokenizer for now like ByT5 etc
- [ ] What effect does the padding/truncating have?
- [x] Audio normalization? Whisper does some normalization based on dataset stats etc
- [x] Positional encoding - currently uses RoPE across the entire embedded sequence of mels + text, should we have separate ones more like encoder-decoder?
- [ ] Masking approach - currently a causal mask is applied across the entire embedded sequence, is something like a prefix mask _substantially_ better?
- [ ] Classifier Free Guidance - seems to be everywhere now in language models - any reason it wouldn't help here?
- [ ] Get HuggingFace streaming and interleaved datasets to work well with buffering/prefetching/multiprocessing - whatever I hacked together initially was at least 10x worse than the usual disk based loading and not usable
- [ ] Scaling data size - Whisper paper notes that performance improves rapidly from 3,000 to 13,000 hours for English
