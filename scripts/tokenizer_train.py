import sentencepiece as spm

import os

model_prefix = "tinyasr"


assert not os.path.exists(
    f"{model_prefix}.model"
), f"Be careful, {model_prefix}.model already exists!"
assert not os.path.exists(
    f"{model_prefix}.vocab"
), f"Be careful, {model_prefix}.vocab already exists!"

spm.SentencePieceTrainer.train(
    input="sentences.txt",
    model_prefix=model_prefix,
    vocab_size=8192,
    byte_fallback=True,
    pad_id=0,
    bos_id=1,
    eos_id=2,
    unk_id=3,
)
