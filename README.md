# tinyasr

Hacking on tiny decoder-only transformers for ASR. This is _very much_ a toy project - I don't have much compute at the moment (just a single 2080 Ti, and maybe the odd cloud VM or Colab 😢), and only have access to the standard and easily attainable ASR datasets (LibriSpeech, Common Voice etc). I'd just like to scratch an itch to see how close to an equivalently sized Whisper model I can get with something like this (and hopefully learn some things along the way too).

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

## Roadmap

- [ ] Make it work
- [ ] Proper benchmarks
- [ ] Text tokenization - just using a utf-8 based tokenizer for now like ByT5 etc
- [ ] What effect does the padding/truncating have? 
- [ ] Audio normalization? Whisper does some normalization based on dataset stats etc
- [ ] Positional encoding - currently uses RoPE across the entire embedded sequence of mels + text, should we have separate ones more like encoder-decoder?
- [ ] Masking approach - currently a causal mask is applied across the entire embedded sequence, is something like a prefix mask _substantially_ better?