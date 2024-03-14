import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from pydantic import BaseModel
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW


class TinyASRConfig(BaseModel):
    # audio
    sample_rate: int = 16_000
    max_duration: float = 15.0
    n_mels: int = 80
    kernel_size: int = 3
    hop_length: int = 160

    # text
    bos_token_id: int = 256
    eos_token_id: int = 256 + 1
    pad_token_id: int = 256 + 1 + 1
    n_tokens: int = 256 + 1 + 1 + 1
    text_max_seqlen: int = 512

    pad_vocab_size_multiple: int = 8
    tie_weights: bool = True

    use_rotary_emb: bool = True

    # decoder
    n_layers: int = 6
    d_model: int = 512
    n_heads: int = 8
    bias: bool = False
    dropout: float = 0.0

    amp_dtype: str = "bfloat16"


class MHA(nn.Module):
    def __init__(self, *, d_model: int, n_heads: int, bias: bool, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: Tensor, rotary_emb: Tensor | None = None):
        qkv = self.Wqkv(x)
        qkv = rearrange(
            qkv, "B T (three h d) -> B three h T d", three=3, h=self.n_heads
        )
        q, k, v = qkv.unbind(dim=1)

        if rotary_emb is not None:
            q = apply_rotary_pos_emb(rotary_emb, q)
            k = apply_rotary_pos_emb(rotary_emb, k)

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout, is_causal=True
        )
        out = self.out_proj(rearrange(out, "... h T d -> ... T (h d)"))

        return out


class Block(nn.Module):
    def __init__(self, *, d_model: int, n_heads: int, dropout: float, bias: bool):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = MHA(
            d_model=d_model,
            n_heads=n_heads,
            bias=bias,
            dropout=dropout,
        )
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, rotary_emb: Tensor | None = None):
        x = x + self.attn(self.attn_norm(x), rotary_emb=rotary_emb)
        x = x + self.mlp(self.mlp_norm(x))
        return x


# lucidrains
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=50000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    @property
    def device(self):
        return self.inv_freq.device

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, t: Tensor):
        t = t.type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@torch.cuda.amp.autocast(enabled=False)
def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        bias: bool,
        use_rotary_emb: bool,
    ):
        super().__init__()

        d_head = d_model // n_heads

        self.use_rotary_emb = use_rotary_emb

        if self.use_rotary_emb:
            self.rotary_emb = RotaryEmbedding(d_head)

        self.blocks = nn.ModuleList(
            [
                Block(d_model=d_model, n_heads=n_heads, dropout=dropout, bias=bias)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor):
        B, T, d = x.size()
        device = x.device

        if self.use_rotary_emb:
            pos_ids = torch.arange(T, device=device)
            rotary_emb = self.rotary_emb(pos_ids)
        else:
            rotary_emb = None

        for block in self.blocks:
            x = block(x, rotary_emb=rotary_emb)

        x = self.norm(x)

        return x


class TinyASR(nn.Module):
    def __init__(self, config: TinyASRConfig):
        super().__init__()

        self.config = config

        kernel_size = config.kernel_size

        self.encoder = nn.Sequential(
            nn.Conv1d(
                config.n_mels,
                config.d_model,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=1,
            ),
            nn.GELU(),
            nn.Conv1d(
                config.d_model,
                config.d_model,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=2,
            ),
            nn.GELU(),
            Rearrange("B d T -> B T d"),
            nn.LayerNorm(config.d_model),
        )

        vocab_size = config.n_tokens
        if vocab_size % config.pad_vocab_size_multiple != 0:
            vocab_size += (
                config.pad_vocab_size_multiple
                - vocab_size % config.pad_vocab_size_multiple
            )

        self.emb = nn.Embedding(vocab_size, config.d_model)

        if not config.use_rotary_emb:
            compression = 2
            audio_max_seqlen = (
                config.max_duration * config.sample_rate // config.hop_length
            ) // compression
            self.audio_pos_emb = nn.Embedding(audio_max_seqlen, config.d_model)

            self.pos_emb = nn.Embedding(config.text_max_seqlen, config.d_model)

        self.decoder = Decoder(
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            dropout=config.dropout,
            bias=config.bias,
            use_rotary_emb=config.use_rotary_emb,
        )

        self.lm_head = nn.Linear(config.d_model, config.n_tokens, bias=config.bias)

        if config.tie_weights:
            self.lm_head.weight = self.emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, mels: Tensor, input_ids: Tensor, target_ids: Tensor | None = None
    ):
        B, C, T = mels.size()
        text_len = input_ids.size(-1)
        device = mels.device

        audio_features = self.encoder(mels)
        audio_len = audio_features.size(1)

        emb = self.emb(input_ids)

        if not self.config.use_rotary_emb:
            text_pos_ids = torch.arange(text_len, device=device)
            audio_pos_ids = torch.arange(audio_len, device=device)

            audio_features = audio_features + self.audio_pos_emb(audio_pos_ids)
            emb = emb + self.pos_emb(text_pos_ids)

        emb = torch.cat((audio_features, emb), dim=1)

        x = self.decoder(emb)

        logits = self.lm_head(x[:, audio_len:])

        if target_ids is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=self.config.pad_token_id,
            )

            return {"loss": loss}
        else:
            return {
                "logits": logits,
            }

    def configure_optimizers(
        self, *, weight_decay: float, lr: float, betas: tuple[float, float]
    ):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, fused=True)

        return optimizer

    @torch.inference_mode()
    def generate(
        self,
        mels: Tensor,
        temperature: float = 1.0,
        top_k: int | None = None,
    ):
        if top_k is not None:
            assert top_k > 0
            assert top_k <= self.config.n_tokens

        B, C, T = mels.size()
        device = mels.device

        mels = self.encoder(mels)

        tokens = torch.full(
            (B, 1), self.config.bos_token_id, dtype=torch.int64, device=device
        )

        for _ in range(self.config.text_max_seqlen):
            x = torch.cat([mels, self.emb(tokens)], dim=1)
            x = self.decoder(x)
            logits = self.lm_head(x[:, -1:])

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            logits = logits.squeeze(1)
            logits[:, self.config.pad_token_id] = -float("inf")
            logits[:, self.config.bos_token_id] = -float("inf")

            probs = F.softmax(logits / temperature, dim=-1)

            token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat((tokens, token), dim=1)

        return tokens


if __name__ == "__main__":
    from .text import detokenize, tokenize

    device = "cuda"

    config = TinyASRConfig()
    model = TinyASR(config).to(device)

    B = 8
    mels = torch.randn(B, config.n_mels, 100, device=device)
    token_ids = torch.randint(0, config.n_tokens, (B, 10), device=device)

    outputs = model(mels, token_ids)

    loss = outputs["loss"]
    loss.backward()

    model.eval()

    generated_token_ids = model.generate(mels)
    generated_texts = detokenize(generated_token_ids)

    print(generated_texts)
