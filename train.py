import torch
from tqdm import tqdm
from utils import AverageMeter
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from model import APTOSModel
from config import Config
from utils import seed_everything, save_checkpoint
from dataset import get_dataloaders
import torch.optim.lr_scheduler


def train_one_epoch(train_loader, model, optimizer, criterion_fn, epoch_idx, writer):
    model.train()
    loss_tracker = AverageMeter()
    num_steps_in_epoch = len(train_loader)
    t_loader = tqdm(train_loader, total=num_steps_in_epoch)
    for b_idx, (imgs, labels) in enumerate(t_loader):
        imgs = imgs.to("cuda", dtype=torch.float)
        labels = labels.to("cuda", dtype=torch.float)

        optimizer.zero_grad()
        pred = model(imgs).squeeze()

        # one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=5)
        loss = criterion_fn(pred, labels)

        loss.backward()
        optimizer.step()
        writer.add_scalar('loss/train', loss.item(), epoch_idx*num_steps_in_epoch + b_idx)
        loss_tracker.update(loss.item())
        t_loader.set_postfix(loss=loss_tracker.avg)
    return loss_tracker.avg


def validate(val_loader, model, criterion_fn):
    loss_tracker = AverageMeter()
    model.eval()
    with torch.no_grad():
        t_loader = tqdm(val_loader, total=len(val_loader))
        for b_idx, data in enumerate(t_loader):
            imgs, labels = data
            imgs = imgs.to("cuda", dtype=torch.float)
            labels = labels.to("cuda", dtype=torch.float)

            pred = model(imgs).squeeze()
            # one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=5)
            loss = criterion_fn(pred, labels)
            loss_tracker.update(loss.item())
    return loss_tracker.avg


def run():
    writer = SummaryWriter(Config.loss_logs_dir)
    seed_everything(seed=42)

    train_loader, val_loader = get_dataloaders(train_df_path=Config.train_path, val_df_path=Config.val_path, img_size=Config.img_size, batch_size=Config.batch_size)

    model = APTOSModel()
    model.cuda()
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 if epoch < 50 else 0.1)
    best_loss = 1000

    for epoch_idx in tqdm(range(Config.epochs)):
        print(f'train epoch {epoch_idx}: ')
        train_one_epoch(train_loader, model, optimizer, criterion, epoch_idx, writer)
        scheduler.step()
        if epoch_idx % 5 == 0:
            print(f'validate epoch {epoch_idx}')
            avg_val_loss = validate(val_loader, model, criterion)
            writer.add_scalar('loss/val', avg_val_loss, epoch_idx)
            if avg_val_loss < best_loss:
                save_checkpoint(Config.checkpoint_dir, 'best_model', model, optimizer, scheduler, epoch_idx)
                best_loss = avg_val_loss
        save_checkpoint(Config.checkpoint_dir, f'{epoch_idx}', model, optimizer, scheduler, epoch_idx)


if __name__ == '__main__':
    run()
