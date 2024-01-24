from pydantic import BaseModel

from .model import TinyASRConfig


class TrainConfig(BaseModel):
    warmup_steps: int = 1_000
    steps: int = 100_000
    min_lr: float = 1e-5
    lr: float = 1e-4

    batch_size: int = 64
    gradient_accumulation_steps: int = 4

    weight_decay: float = 0.1
    max_norm: float = 1.0
    betas: tuple[float, float] = (0.9, 0.95)

    val_every: int = 5_000
    val_items: int = 128

    checkpoint_every: int = 5_000
    log_every: int = 10

    num_workers: int = 4
    seed: int = 42

    compile: bool = False

    dataset: str = "common_voice"

    checkpoint: str | None = None # TODO checkpoint to restart training from

    @property
    def val_steps(self) -> int:
        # always drop_last in DataLoader
        return self.val_items // self.batch_size


class Config(BaseModel):
    model: TinyASRConfig = TinyASRConfig()
    train: TrainConfig = TrainConfig()
