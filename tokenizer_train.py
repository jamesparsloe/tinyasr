import sentencepiece as spm


spm.SentencePieceTrainer.train(
    input="sentences.txt",
    model_prefix="tinyasr",
    vocab_size=8192,
    byte_fallback=True,
    pad_id=0,
    bos_id=1,
    eos_id=2,
    unk_id=3,
)
