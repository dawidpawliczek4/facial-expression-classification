import os
import torch
import torch.nn as nn
import numpy as np
import wandb
from facexpr.data.load_data import make_dataloaders
from facexpr.vision_transformer.model import VisionTransformer
from facexpr.utils.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import CosineAnnealingLR
from facexpr.utils.visualization import plot_confusion_matrix, plot_f1_history
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from torch.amp import autocast, GradScaler
from torch.optim import AdamW

CONFIG = {
    "learning_rate": 1e-4,
    "num_classes": 7,
    "patch_size": 8,
    "img_size": 48,
    "in_channels": 1,
    "num_heads": 8,
    "dropout": 1e-3,
    "hidden_dim": 3 * 48,
    "weight_decay": 0,
    "betas": (0.9, 0.999),
    "activation": "gelu",
    "num_encoders": 2,
    "embed_dim": (8 ** 2) * 1, # (CONFIG["patch_size"] ** 2) * CONFIG["in_channels"] = 64
    "num_patches": (48 // 8) ** 2, # (CONFIG["img_size"] // CONFIG["patch_size"]) ** 2 = 36
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 64,
    "epochs": 30,
    "data_dir": "./data/downloaded_data/data",
    "save_path": "./outputs/models/model.pth",
    "project": "fer2013-vision-transformer",
    "name": "1-ViT-patch:8/heads:8/encoders:2"
}

# CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = True

def evaluate_model(model, dataloader, device, criterion):
    model.eval()
    all_preds, all_labels = [], []
    val_loss = val_correct = val_total = 0
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

def train():
    wandb.init(project=CONFIG["project"], config=CONFIG, name=CONFIG["name"])
    os.makedirs(os.path.dirname(CONFIG["save_path"]), exist_ok=True)
    device = torch.device(CONFIG["device"])

    class_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    f1_history = []

    print("Loading data...")
    loaders = make_dataloaders(
        data_dir=CONFIG["data_dir"],
        batch_size=CONFIG["batch_size"],
        img_size=CONFIG["img_size"],
        num_workers=2,
        augment=True,
        grayscale=False
    )
    train_loader, val_loader = loaders["train"], loaders["val"]

    model = VisionTransformer(CONFIG).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        betas=CONFIG["betas"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    scaler = GradScaler()

    early_stopping = EarlyStopping(
        patience=CONFIG["patience"] if "patience" in CONFIG else 5,
        min_delta=CONFIG["min_delta"] if "min_delta" in CONFIG else 0.001,
        monitor=CONFIG["monitor"] if "monitor" in CONFIG else "val_loss"
    )

    print("Starting loop...")
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        train_loss = 0.0
        train_correct = train_total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast('cuda', dtype=torch.bfloat16):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=False)
            scaler.step(optimizer)
            scaler.update()

            preds = outputs.argmax(dim=1)
            train_loss += loss.item() * imgs.size(0)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()

        train_loss /= train_total
        train_acc = train_correct / train_total

        cm, f1, report, val_loss, val_correct, val_total = evaluate_model(model, val_loader, device, criterion)
        val_loss /= val_total
        val_acc = val_correct / val_total

        f1_history.append(f1)
        plot_confusion_matrix(cm, epoch, class_names)

        wandb.log({
            f"classification_report/epoch_{epoch}": wandb.Table(
                data=[[line] for line in report.split("\n")],
                columns=["report"]
            )
        })

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "val/f1": f1
        })

        print(f"Epoch {epoch:02d} | Train loss {train_loss:.4f}, acc {train_acc:.4f} | Val loss {val_loss:.4f}, acc {val_acc:.4f}")

        metrics = {"val_loss": val_loss, "val_acc": val_acc, "val_f1": f1}
        if early_stopping(model, metrics):
            print(f"Early stopping triggered after {epoch} epochs")
            break

    early_stopping.load_best_model(model)
    
    f1_history_np = np.stack(f1_history)
    plot_f1_history(f1_history_np, class_names)

    torch.save(model.state_dict(), CONFIG["save_path"])
    print(f"Best model saved to {CONFIG['save_path']}")

if __name__ == "__main__":
    train()