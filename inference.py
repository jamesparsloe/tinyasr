from tinyasr.model import TinyASR
import sentencepiece as spm
import torch

model = TinyASR.from_pretrained("./runs/z305qf44/tinyasr-005000.pt").eval().cuda()
tokenizer = spm.SentencePieceProcessor(model_file="tinyasr.model")

prompt = "The quick brown"
prompt = "It was a bright cold day "
prompt = "2 + 2 = 4. 3 + 3 = "

device = "cuda"

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

    print(tokenizer.decode(generated_token_ids[0, :length].cpu().tolist()))
