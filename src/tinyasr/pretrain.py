import math
import os
import time
from contextlib import nullcontext
from functools import partial
from itertools import islice

import click
import nltk
import torch
import torch.nn.functional as F
import torchaudio
import yaml
from datasets import interleave_datasets, load_dataset
from nltk import sent_tokenize
from nltk.tokenize import sent_tokenize
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from torchdata.dataloader2.reading_service import (
    DistributedReadingService,
    MultiProcessingReadingService,
)
from torchdata.datapipes.iter import HuggingFaceHubReader

import wandb

from .config import Config
from .model import TinyASR, TinyASRConfig
from .text import ByteLevelTokenizer, SentencePieceTokenizer
from .utils import cycle, warmup_then_cosine_decay

CACHE_DIR = os.path.expanduser("~/.cache/torchaudio")
os.makedirs(CACHE_DIR, exist_ok=True)


nltk.download("punkt")


def text_iter(ds: IterableDataset, max_text_len: int = 128):
    for item in ds:
        text = item["text"]
        for sentence in sent_tokenize(text):
            if sentence < max_text_len:
                yield sentence


def collate(
    batch,
    *,
    tokenizer,
    text_pad_or_truncate: int = 128,
):
    texts = []

    for text in batch:
        texts.append(text)

    token_ids = tokenizer.encode(
        texts,
        pad_or_truncate=text_pad_or_truncate,
    )

    return {
        "token_ids": token_ids,
        "texts": texts,
    }


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--edit", is_flag=True)
def main(config_path: str, edit: bool):
    assert os.getenv("WANDB_API_KEY"), "Please set WANDB_API_KEY"
    assert os.getenv("HF_TOKEN"), "Please set HF_TOKEN"

    name = "tinyasr"
    device = "cuda"

    with open(config_path) as f:
        s = f.read()
        if edit:
            s = click.edit(s)
        config = Config(**yaml.safe_load(s))

    train_config = config.train
    model_config = config.model

    assert model_config.text_pretrain

    run = wandb.init(project=name, config=config.model_dump())

    run_dir = os.path.join("./runs", run.id)
    os.makedirs(run_dir, exist_ok=True)

    if model_config.tokenizer == "byte-level":
        tokenizer = ByteLevelTokenizer(
            pad_token_id=model_config.pad_token_id,
            bos_token_id=model_config.bos_token_id,
            eos_token_id=model_config.eos_token_id,
        )
    else:
        tokenizer = SentencePieceTokenizer(model_config.tokenizer)
        model_config.n_tokens = tokenizer._model.vocab_size()

    _collate = partial(collate, tokenizer=tokenizer)

    batch_size = train_config.batch_size

    dp = (
        HuggingFaceHubReader("mc4", name="en", split="train", trust_remote_code=True)
        .map(lambda x: x["text"])
        .flatmap(sent_tokenize)
        .filter(lambda x: len(x) < model_config.max_text_len)
        .shuffle()
        .batch(batch_size, drop_last=True)
        .collate(_collate)
        .cycle()
    )

    rs = MultiProcessingReadingService(num_workers=4)
    dl = DataLoader2(dp, reading_service=rs)

    dl_iter = iter(dl)

    model = TinyASR(model_config).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"{n_params / 1e6:.1f}M params")

    optimizer = model.configure_optimizers(
        weight_decay=train_config.weight_decay,
        lr=train_config.lr,
        betas=train_config.betas,
    )

    if train_config.compile:
        print(f"Compiling model...")
        model = torch.compile(model)

    step = 0

    t1 = time.perf_counter()

    get_lr = partial(
        warmup_then_cosine_decay,
        warmup_steps=train_config.warmup_steps,
        steps=train_config.steps,
        min_lr=train_config.min_lr,
        max_lr=train_config.lr,
    )

    amp_dtype = (
        torch.bfloat16 if model_config.amp_dtype == "bfloat16" else torch.float32
    )
    ctx = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        if amp_dtype == torch.bfloat16
        else nullcontext()
    )

    while step < train_config.steps:
        lr = get_lr(step)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        for micro_step in range(train_config.gradient_accumulation_steps):
            batch = next(dl_iter)

            token_ids = batch["token_ids"].to(device, non_blocking=True)

            with ctx:
                out = model(token_ids=token_ids)

            loss = out["loss"] / train_config.gradient_accumulation_steps
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_config.max_norm
        )

        optimizer.step()
        optimizer.zero_grad()

        step += 1

        if step % train_config.log_every == 0:
            t2 = time.perf_counter()
            samples = (
                train_config.batch_size
                * train_config.gradient_accumulation_steps
                * train_config.log_every
            )
            throughput = samples / (t2 - t1)
            wandb.log(
                {
                    "train/loss": loss.item()
                    * train_config.gradient_accumulation_steps,
                    "train/grad_norm": grad_norm.item(),
                    "train/throughput": throughput,
                    "train/lr": lr,
                },
                step=step,
            )
            t1 = t2

        if step % train_config.val_every == 0:
            print(f"Starting val")
            model.eval()

            data = []

            for prompt_text in [
                "Hello, ",
                "The quick brown ",
                "2 + 2 =",
                "It was a bright cold day ",
                "The river flows",
            ]:
                # prompt_token_ids = tokenizer.encode([prompt_text])
                prompt_token_ids = torch.tensor(
                    [tokenizer._model.bos_id()] + tokenizer._model.encode(prompt_text),
                    device=device,
                )
                prompt_token_ids = prompt_token_ids.unsqueeze(0)

                with ctx:
                    generated_token_ids = model.generate_unconditional(prompt_token_ids)

                eos_mask = (generated_token_ids == tokenizer._model.eos_id()).float()
                before_eos_mask = eos_mask.cumsum(dim=-1) == 0
                length = before_eos_mask.sum(dim=-1).item()

                generated_text = tokenizer._model.decode(
                    generated_token_ids[0, :length].cpu().tolist()
                )

                data.append([prompt_text, generated_text])

            eval_table = wandb.Table(columns=["prompt", "generated"], data=data)
            wandb.log({"eval": eval_table}, step=step)

            model.train()

        if step % train_config.checkpoint_every == 0:
            checkpoint = {
                "step": step,
                "config": config.model_dump(),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            checkpoint_path = os.path.join(run_dir, f"{name}-{step:06d}.pt")

            print(f"Saved checkpoint to {checkpoint_path}")

            torch.save(checkpoint, checkpoint_path)


if __name__ == "__main__":
    main()
