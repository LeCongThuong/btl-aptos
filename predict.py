from model import APTOSModel
from dataset import get_dataloaders
import pandas as pd
import torch
from tqdm.auto import tqdm
from config import Config
from utils import load_checkpoint


def predict(val_loader, model):
    model.eval()
    pred_list = []
    with torch.no_grad():
        t_loader = tqdm(val_loader, total=len(val_loader))
        for b_idx, data in enumerate(t_loader):
            imgs, labels = data
            imgs = imgs.cuda()
            pred = model(imgs)
            _, predicted = torch.max(pred.data,  1)
            pred_list.extend(predicted.tolist())
    return pred_list


def run():
    checkpoint_dir = "logs/checkpoints"
    name = 'best_model'
    model = APTOSModel()
    model, _, _ = load_checkpoint(checkpoint_dir, name, model)
    model.cuda()
    _, val_loader = get_dataloaders(train_df_path=Config.train_path, val_df_path=Config.val_path, img_size=Config.img_size, batch_size=Config.batch_size)
    pred_list = predict(val_loader, model)
    val_df = pd.read_csv(Config.val_path)
    val_df['pred'] = pred_list
    val_df.to_csv(f'./logs/val_res/model_pred_{name}.csv')


if __name__ == '__main__':
    run()
