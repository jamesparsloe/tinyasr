import jiwer
import os
from tinyasr.model import TinyASR, TinyASRConfig
from tinyasr.train import collate
from datasets import load_dataset
from tinyasr.text import ByteLevelTokenizer, SentencePieceTokenizer
from functools import partial
from torch.utils.data import DataLoader
import torchaudio
import torch
import torch.nn.functional as F
import torchaudio
import yaml
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence

# pip install openai-whisper
from whisper.normalizers import EnglishTextNormalizer


def main():
    device = "cuda"
    amp_dtype = torch.bfloat16

    normalizer = EnglishTextNormalizer()

    ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype)

    runs_dirs = "./runs"
    mel_transform = torchaudio.transforms.MelSpectrogram().to(device)

    for run_id in os.listdir(runs_dirs):
        run_dir = os.path.join(runs_dirs, run_id)

        files = os.listdir(run_dir)
        paths = [os.path.join(run_dir, basename) for basename in files]

        if len(paths) > 0:
            checkpoint_path = max(paths, key=os.path.getctime)
            print(run_dir, checkpoint_path)

            model = TinyASR.from_pretrained(checkpoint_path).eval().to(device)
            model_config = model.config

            if model_config.tokenizer == "byte-level":
                tokenizer = ByteLevelTokenizer(
                    pad_token_id=model_config.pad_token_id,
                    bos_token_id=model_config.bos_token_id,
                    eos_token_id=model_config.eos_token_id,
                )
            else:
                tokenizer = SentencePieceTokenizer(model_config.tokenizer)
                model_config.n_tokens = tokenizer._model.vocab_size()

            ds = load_dataset("librispeech_asr", "clean", split="test")

            print(len(ds))

            num_workers = 4
            batch_size = 64

            collate_fn = partial(
                collate,
                tokenizer=tokenizer,
                sample_rate=model_config.sample_rate,
                audio_pad_or_truncate=model_config.max_duration,
                text_pad_or_truncate=model_config.max_text_len,
            )
            dl = DataLoader(
                ds,
                shuffle=True,
                batch_size=batch_size,
                num_workers=num_workers,
                drop_last=True,
                collate_fn=collate_fn,
            )

            references = []
            predictions = []

            for batch in dl:
                texts = batch["texts"]
                waveforms = batch["waveforms"].to(device, non_blocking=True)
                mels = mel_transform(waveforms)
                with ctx:
                    generated_token_ids = model.generate(mels)

                generated_texts = tokenizer.decode(generated_token_ids)

                references.extend(texts)
                predictions.extend(generated_texts)

            references = [normalizer(text) for text in references]
            predictions = [normalizer(text) for text in predictions]

            wer = jiwer.wer(reference=references, hypothesis=predictions)

            print(f"{run_id=} {wer=}")


if __name__ == "__main__":
    main()
