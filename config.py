from dataclasses import dataclass


@dataclass
class Config:
    train_path: str = './data/train.csv'
    val_path: str = './data/val.csv'
    epochs: int = 60
    lr: float = 0.0002
    img_size: int = 512
    batch_size: int = 32
    checkpoint_dir: str = './logs/checkpoints'
    loss_logs_dir: str = './logs/loss'
