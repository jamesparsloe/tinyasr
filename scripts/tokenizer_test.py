import sentencepiece as spm


s = spm.SentencePieceProcessor(model_file="tinyasr.model")

print(s.pad_id())
print(s.bos_id())
print(s.eos_id())

text = (
    "Curly Bill's second-in-command, Johnny Ringo, becomes the new head of the Cowboys."
)

encoded = s.encode(text)

print(f"{len(text)=} -> {len(encoded)=}")
print(encoded)

print(s.decode(s.encode("Hello, world!")))
