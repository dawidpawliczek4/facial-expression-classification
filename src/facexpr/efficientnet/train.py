import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from facexpr.data.load_data import make_dataloaders
from facexpr.efficientnet.model import EfficientNetV2Classifier
from facexpr.efficientnet.model_simple import SimpleClassifier
from torch.optim.lr_scheduler import CosineAnnealingLR
from facexpr.utils.visualization import plot_confusion_matrix, plot_f1_history
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from torch.amp import autocast, GradScaler
from torch.optim import AdamW

CONFIG = {
    "data_dir": "./data/downloaded_data/data",
    "batch_size": 32,
    "epochs": 10,
    "lr": 1e-3,
    "save_path": "./outputs/models/model.pth",
    "img_size": 224,
    "project": "fer2013-efficientnetv2",
    "name": "test-simpleclassifier"
}

def main():
    wandb.init(project=CONFIG["project"], config=CONFIG, name=CONFIG["name"])

    os.makedirs(os.path.dirname(CONFIG["save_path"]), exist_ok=True)

    class_names = ["Angry", "Disgust", "Fear",
                   "Happy", "Sad", "Surprise", "Neutral"]
    f1_history = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("loading data...")
    loaders = make_dataloaders(
        data_dir=CONFIG["data_dir"],
        batch_size=CONFIG["batch_size"],
        img_size=CONFIG['img_size'],
        num_workers=2,
        augment=True,
        grayscale=False
    )
    train_loader, val_loader = loaders["train"], loaders["val"]

    model = SimpleClassifier(num_classes=7).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = AdamW(model.parameters(), lr=CONFIG["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    scaler = GradScaler()

    print("starting loop...")
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        train_loss = 0.0
        train_correct = train_total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast('cuda'):
                outputs = model(imgs, labels)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
                error_if_nonfinite=False
            )
            scaler.step(optimizer)
            scaler.update()        

            with torch.no_grad():
                raw_logits = model(imgs)
            preds = raw_logits.argmax(dim=1)

            train_loss += loss.item() * imgs.size(0)            
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()
        train_loss /= train_total
        train_acc = train_correct / train_total

        cm, f1, report, val_loss, val_correct, val_total = evaluate_model(
            model, val_loader, device, criterion)
        val_loss /= val_total
        val_acc = val_correct / val_total

        f1_history.append(f1)
        plot_confusion_matrix(cm, epoch, class_names)

        wandb.log({
            f"classification_report/epoch_{epoch}": wandb.Table(
                data=[[line] for line in report.split("\n")], columns=["report"])})

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
        })

        print(f"Epoch {epoch:02d} | "
              f"Train loss {train_loss:.4f}, acc {train_acc:.4f} | "
              f"Val loss {val_loss:.4f}, acc {val_acc:.4f}")

    f1_history_np = np.stack(f1_history)
    plot_f1_history(f1_history_np, class_names)

    torch.save(model.state_dict(), CONFIG["save_path"])
    print(f"Model saved to {CONFIG['save_path']}")


def evaluate_model(model, dataloader, device, criterion):
    model.eval()
    all_preds = []
    all_labels = []
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)

            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    report = classification_report(y_true, y_pred, digits=3, zero_division=0)
    return cm, f1, report, val_loss, val_correct, val_total


if __name__ == "__main__":
    main()
