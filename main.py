import os
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

from src.datasets import ThingsMEGDataset
from src.models import MEGNet  # MEGNetを使用するように更新
from src.utils import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    print("---------------------------------start_dataload--------------------------------------")
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    
    train_set = ThingsMEGDataset("train", args.data_dir)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset("val", args.data_dir)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, **loader_args)
    print("----------------------------------end_dataload----------------------------------------")

    num_subjects = train_set.num_subjects  # Assuming this method is available in ThingsMEGDataset
    model = MEGNet(
        num_classes=train_set.num_classes,
        seq_len=train_set.seq_len,
        in_channels=train_set.num_channels,
        num_subjects=num_subjects
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    max_val_acc = 0
    accuracy = Accuracy(task="multiclass", num_classes=train_set.num_classes, top_k=10).to(args.device)
      
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
        model.train()

        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            X, y = X.float().to(args.device), y.to(args.device)
            subject_idxs = subject_idxs.to(args.device)
            
            y_pred = model(X, subject_idxs)
            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        model.eval()
        
        with torch.no_grad():
            for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
                X, y = X.float().to(args.device), y.to(args.device)
                subject_idxs = subject_idxs.to(args.device)
                
                y_pred = model(X, subject_idxs)
                val_loss.append(F.cross_entropy(y_pred, y).item())
                val_acc.append(accuracy(y_pred, y).item())

        epoch_stats = f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}"
        print(epoch_stats)
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})

        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)

    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))
    model.eval()
    preds = []
    for X, subject_idxs in tqdm(test_loader, desc="Test"):
        X = X.float().to(args.device)
        subject_idxs = subject_idxs.to(args.device)
        preds.append(model(X, subject_idxs).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()
