import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


def tokenize(
    texts: list[str],
    bos_token_id: int = 256,
    eos_token_id: int = 256 + 1,
    pad_token_id: int = 256 + 1 + 1,
):
    batch = []
    for text in texts:
        token_ids = torch.tensor(list(text.encode("utf-8")))
        token_ids = F.pad(token_ids, (1, 0), value=bos_token_id)
        token_ids = F.pad(token_ids, (0, 1), value=eos_token_id)
        batch.append(token_ids)

    batch = pad_sequence(batch, batch_first=True, padding_value=pad_token_id)

    return batch


def detokenize(
    batch: Tensor,
    bos_token_id: int = 256,
    eos_token_id: int = 256 + 1,
    pad_token_id: int = 256 + 1 + 1,
):
    texts = []
    for token_ids in batch:
        token_ids = token_ids[token_ids != pad_token_id]
        token_ids = token_ids[token_ids != bos_token_id]

        eos_mask = (token_ids == eos_token_id).float()
        before_eos_mask = eos_mask.cumsum(dim=-1) == 0
        length = before_eos_mask.sum(dim=-1).item()

        texts.append("".join(chr(token_id) for token_id in token_ids[:length]))

    return texts


if __name__ == "__main__":
    texts = ["Does this work?", "Hello?"]
    tokenized = tokenize(texts)

    print(tokenized)
    print(len(tokenized[0]))

    detokenized = detokenize(tokenized)

    for text, d in zip(texts, detokenized):
        print(f"{text} -> {d}")
        assert text == d
