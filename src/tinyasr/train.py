import math
import os
import time
from contextlib import nullcontext
from functools import partial
from itertools import islice

import click
import torch
import torch.nn.functional as F
import torchaudio
import yaml
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService

import wandb

from .audio import WhisperMelSpectrogram
from .config import Config
from .data import build_dp
from .model import TinyASR, TinyASRConfig
from .text import detokenize, tokenize
from .utils import seed_all

CACHE_DIR = os.path.expanduser("~/.cache/torchaudio")
os.makedirs(CACHE_DIR, exist_ok=True)


def collate(
    batch,
    *,
    sample_rate: int,
    max_duration: float,
):
    texts = []
    waveforms = []

    audio_len = int(max_duration * sample_rate)

    for item in batch:
        waveform = item["waveform"]
        pad = audio_len - waveform.size(-1)
        waveform = F.pad(waveform, (0, pad), value=0.0)
        waveforms.append(waveform)
        texts.append(item["text"])

    waveforms = torch.stack(waveforms)
    token_ids = tokenize(texts)

    return {
        "waveforms": waveforms,
        "token_ids": token_ids,
        "texts": texts,
    }


def cycle(iterable):
    while True:
        for item in iterable:
            yield item


def warmup_then_cosine_decay(
    step: int, *, warmup_steps: int, steps: int, min_lr: float, max_lr: float
):
    if step < warmup_steps:
        return min_lr + step * (max_lr - min_lr) / (warmup_steps)
    elif step > steps:
        return min_lr
    else:
        decay_ratio = (step - warmup_steps) / (steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def main(config_path: str):
    assert os.getenv("WANDB_API_KEY"), "Please set WANDB_API_KEY"
    assert os.getenv("HF_TOKEN"), "Please set HF_TOKEN"

    name = "tinyasr"
    device = "cuda"

    with open(config_path) as f:
        config = Config(**yaml.safe_load(f))

    train_config = config.train
    model_config = config.model

    run = wandb.init(project=name, config=config.model_dump())

    seed_all(train_config.seed)

    run_dir = os.path.join("./runs", run.id)
    os.makedirs(run_dir, exist_ok=True)

    collate_fn = partial(
        collate,
        sample_rate=model_config.sample_rate,
        max_duration=model_config.max_duration,
    )

    train_dp = (
        build_dp(split="train", max_duration=model_config.max_duration)
        .cycle()
        .shuffle(buffer_size=1000)
        .batch(train_config.micro_batch_size, drop_last=True)
        .collate(collate_fn)
    )

    val_dp = (
        build_dp(split="val", max_duration=model_config.max_duration)
        .header(train_config.val_items)
        .batch(train_config.micro_batch_size, drop_last=True)
        .collate(collate_fn)
    )

    num_workers = (os.cpu_count() - 1) // 2

    # rs = MultiProcessingReadingService(num_workers=num_workers)
    rs = None
    train_dl = DataLoader2(train_dp, reading_service=rs)
    train_dl_iter = iter(train_dl)
    val_dl = DataLoader2(val_dp, reading_service=rs)

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

    mel_transform = WhisperMelSpectrogram().to(device)

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

    gradient_accumulation_steps = train_config.gradient_accumulation_steps

    batch = next(train_dl_iter)

    while step < train_config.steps:
        lr = get_lr(step)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        for micro_step in range(gradient_accumulation_steps):
            waveforms = batch["waveforms"].to(device, non_blocking=True)
            token_ids = batch["token_ids"].to(device, non_blocking=True)
            input_ids, target_ids = token_ids[:, :-1], token_ids[:, 1:].contiguous()

            mels = mel_transform(waveforms)

            with ctx:
                out = model(mels, input_ids, target_ids=target_ids)

            batch = next(train_dl_iter)

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
                    "train/loss": gradient_accumulation_steps * loss.item(),
                    "train/grad_norm": grad_norm.item(),
                    "train/throughput": throughput,
                    "train/lr": lr,
                },
                step=step,
            )
            t1 = t2

            print(f"{step=} {throughput=:.1f} {loss.item()=}")

        if step % train_config.val_every == 0:
            print(f"Starting val")
            model.eval()

            data = []

            for val_batch in val_dl:
                texts = val_batch["texts"]
                waveforms = val_batch["waveforms"].to(device, non_blocking=True)
                mels = mel_transform(waveforms)

                with ctx:
                    generated_token_ids = model.generate(mels)

                generated_texts = detokenize(generated_token_ids)

                for text, generated_text in zip(texts, generated_texts):
                    data.append([text, generated_text])

            eval_table = wandb.Table(columns=["target", "prediction"], data=data)
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
