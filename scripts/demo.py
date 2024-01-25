import gradio as gr
import torchaudio
import torch
from tinyasr.model import TinyASR, TinyASRConfig
import torch.nn.functional as F
from tinyasr.text import tokenize, detokenize
from contextlib import nullcontext

device = "cuda"


checkpoint_path = "./runs/7srpf3at/tinyasr-100000.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
config = TinyASRConfig(**checkpoint["config"])
model = TinyASR(config)
state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()}
model.load_state_dict(state_dict)
model = model.to(device).eval()

mel_transform = torchaudio.transforms.MelSpectrogram().to(device)


def transcribe(path: str, temperature: float):
    waveform, sample_rate = torchaudio.load(path)
    waveform = torchaudio.functional.resample(waveform, sample_rate, config.sample_rate)
    audio_len = int(config.max_duration * config.sample_rate)
    pad = audio_len - waveform.size(-1)
    waveform = F.pad(waveform, (0, pad))
    waveform = waveform.to(device)
    mel = mel_transform(waveform)

    amp_dtype = (
        torch.bfloat16 if model.config.amp_dtype == "bfloat16" else torch.float32
    )
    ctx = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        if amp_dtype == torch.bfloat16
        else nullcontext()
    )

    with ctx:
        token_ids = model.generate(mel, temperature=temperature)

    print(f"{token_ids=}")

    text = detokenize(token_ids)
    text = text[0]

    return text


demo = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(type="filepath"),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=1.0, label="temperature"),
    ],
    outputs=gr.Textbox(),
)


demo.launch(debug=True)
