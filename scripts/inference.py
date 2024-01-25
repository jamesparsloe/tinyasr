from tinyasr.model import TinyASR
import sentencepiece as spm
import torch

checkpoint_path = "./runs/lj70b56v/tinyasr-060000.pt"
model = TinyASR.from_pretrained(checkpoint_path).eval().cuda()
tokenizer = spm.SentencePieceProcessor(model_file="tinyasr.model")
device = "cuda"

for prompt in [
    "The quick brown ",
    "It was a bright cold day ",
    "2 + 2 = 4. 2 + 3 = 5. 3 + 3 = ",
    "Does this work? I",
    "Red ",
    "Once upon a time there ",
    "The meaning of life is "
]:
    with torch.inference_mode(), torch.amp.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        token_ids = torch.tensor(
            [tokenizer.bos_id()] + tokenizer.encode(prompt), device=device
        )
        token_ids = token_ids.unsqueeze(0)
        generated_token_ids = model.generate_unconditional(token_ids, temperature=0.3)

        eos_mask = (generated_token_ids == tokenizer.eos_id()).float()
        before_eos_mask = eos_mask.cumsum(dim=-1) == 0
        length = before_eos_mask.sum(dim=-1).item()

        generated = tokenizer.decode(generated_token_ids[0, :length].cpu().tolist())

        print(f"{prompt} -> {generated}")
