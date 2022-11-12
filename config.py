from dataclasses import dataclass


@dataclass
class Config:
    train_path: str = '/mnt/hdd/thuonglc/study/btl-aptos/data/train.csv'
    val_path: str = '/mnt/hdd/thuonglc/study/btl-aptos/data/val.csv'
    epochs: int = 60
    lr: float = 0.0002
    img_size: int = 512
    batch_size: int = 32
    checkpoint_dir: str = '/mnt/hdd/thuonglc/study/btl-aptos/logs/checkpoints'
    loss_logs_dir: str = '/mnt/hdd/thuonglc/study/btl-aptos/logs/loss'
