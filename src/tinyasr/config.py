from pydantic import BaseModel

from .model import TinyASRConfig


class TrainConfig(BaseModel):
    warmup_steps: int = 1000
    steps: int = 100_000
    min_lr: float = 5e-5
    lr: float = 5e-4

    batch_size: int = 64
    micro_batch_size: int = 64

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.batch_size // self.micro_batch_size

    weight_decay: float = 0.1
    max_norm: float = 1.0
    betas: tuple[float, float] = (0.9, 0.95)

    val_every: int = 5_000
    val_items: int = 128

    checkpoint_every: int = 5_000
    log_every: int = 10

    num_workers: int = 4
    seed: int = 42

    shuffle_buffer_size: int = 256

    compile: bool = False

    @property
    def val_steps(self) -> int:
        # always drop_last in DataLoader
        return self.val_items // self.batch_size


class Config(BaseModel):
    model: TinyASRConfig = TinyASRConfig()
    train: TrainConfig = TrainConfig()
