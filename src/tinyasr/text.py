import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


def tokenize(texts: list[str], bos_token_id: int = 0, eos_token_id: int = 1):
    n_special_tokens = 2
    batch = []
    for text in texts:
        token_ids = torch.tensor(list(text.encode("utf-8"))) + n_special_tokens
        token_ids = F.pad(token_ids, (1, 0), value=bos_token_id)
        token_ids = F.pad(token_ids, (0, 1), value=eos_token_id)
        batch.append(token_ids)

    batch = pad_sequence(batch, batch_first=True, padding_value=eos_token_id)

    return batch


def detokenize(batch: Tensor, bos_token_id: int = 0, eos_token_id: int = 1):
    n_special_tokens = 2
    texts = []
    for token_ids in batch:
        token_ids = token_ids[token_ids != eos_token_id]
        token_ids = token_ids[token_ids != bos_token_id]
        token_ids = token_ids - n_special_tokens

        texts.append("".join(chr(token_id) for token_id in token_ids))

    return texts


if __name__ == "__main__":
    texts = ["Does this work?", "Hello?"]
    tokenized = tokenize(texts)
    detokenized = detokenize(tokenized)

    print(tokenized)

    for text, d in zip(texts, detokenized):
        print(f"{text} -> {d}")
        assert text == d
