import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


class ByteLevelTokenizer:
    def __init__(
        self,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ):
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.n_special_tokens = 1 + 1 + 1

    def encode(
        self,
        texts: list[str],
        pad_or_truncate: int = 128,
    ):
        batch = []
        for text in texts:
            token_ids = torch.tensor(list(text.encode("utf-8"))) + self.n_special_tokens
            token_ids = token_ids[: pad_or_truncate - self.n_special_tokens]
            token_ids = F.pad(token_ids, (1, 0), value=self.bos_token_id)
            token_ids = F.pad(token_ids, (0, 1), value=self.eos_token_id)
            pad = pad_or_truncate - token_ids.size(-1)
            token_ids = F.pad(token_ids, (0, pad), value=self.pad_token_id)
            batch.append(token_ids)

        batch = pad_sequence(batch, batch_first=True, padding_value=self.pad_token_id)

        return batch

    def decode(
        self,
        batch: Tensor,
    ):
        n_special_tokens = 1 + 1 + 1
        texts = []
        for token_ids in batch:
            token_ids = token_ids[token_ids != self.pad_token_id]
            token_ids = token_ids[token_ids != self.bos_token_id]

            eos_mask = (token_ids == self.eos_token_id).float()
            before_eos_mask = eos_mask.cumsum(dim=-1) == 0
            length = before_eos_mask.sum(dim=-1).item()

            token_ids = token_ids - n_special_tokens
            texts.append("".join(chr(token_id) for token_id in token_ids[:length]))

        return texts


class SentencePieceTokenizer:
    def __init__(self, path: str):
        self._model = spm.SentencePieceProcessor(model_file=path)

    def encode(
        self,
        texts: list[str],
        pad_or_truncate: int = 128,
    ):
        batch = []
        for text in texts:
            token_ids = torch.tensor(
                [self._model.bos_id()]
                + self._model.encode(text)
                + [self._model.eos_id()]
            )
            token_ids = token_ids[:pad_or_truncate]
            pad = pad_or_truncate - token_ids.size(-1)
            token_ids = F.pad(token_ids, (0, pad), value=self._model.pad_id())
            batch.append(token_ids)

        batch = pad_sequence(
            batch, batch_first=True, padding_value=self._model.pad_id()
        )

        return batch

    def decode(
        self,
        batch: Tensor,
    ):
        texts = []
        for token_ids in batch:
            token_ids = token_ids[token_ids != self._model.pad_id()]
            token_ids = token_ids[token_ids != self._model.bos_id()]

            eos_mask = (token_ids == self._model.eos_id()).float()
            before_eos_mask = eos_mask.cumsum(dim=-1) == 0
            length = before_eos_mask.sum(dim=-1).item()

            token_ids = token_ids[:length]
            texts.append(self._model.decode(token_ids.tolist()))

        return texts


if __name__ == "__main__":
    tokenizer = ByteLevelTokenizer()
    tokenizer = SentencePieceTokenizer("tinyasr.model")
    print(tokenizer._model.vocab_size())
    texts = ["Does this work?", "Hello?"]
    tokenized = tokenizer.encode(texts)
    detokenized = tokenizer.decode(tokenized)

    print(tokenized)

    for text, d in zip(texts, detokenized):
        print(f"{text} -> {d}")
        assert text == d
