import os
import torch
import torch.nn as nn
import numpy as np
import wandb
import matplotlib.pyplot as plt
from facexpr.data.load_data_albumentations import make_dataloaders
from facexpr.efficientnet.model import EfficientNetV2Classifier
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from facexpr.utils.visualization import plot_confusion_matrix, plot_f1_history
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from facexpr.utils.early_stopping import EarlyStopping

CONFIG = {
    "data_dir": "./data/downloaded_data/data",
    "batch_size": 128,
    "epochs": 50,
    "lr": 5e-3,
    "save_path": "./outputs/models/model.pth",
    "img_size": 224,
    "project": "fer2013-efficientnetv2",
    "name": "19-better-augment",
    "lr-cbam": 5e-2,

    # Early stopping parameters
    "patience": 7,
    "min_delta": 0.001,
    "monitor": "val_loss",
}

# CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)


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
    visualize_shhit(train_loader, class_names)

    model = EfficientNetV2Classifier(num_classes=7).to(device)

    def is_cbam_mhsa(name):
        res = any(tag in name.lower() for tag in [
                  "cbam", "channel_att", "spatial_att", "self_attention", "mhsa", "attention", "att"])
        return res

    cbam_params = [p for n, p in model.named_parameters() if is_cbam_mhsa(n)]
    other_params = [p for n, p in model.named_parameters()
                    if not is_cbam_mhsa(n)]

    total_steps = CONFIG["epochs"] * len(train_loader)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(other_params, lr=CONFIG["lr"], weight_decay=1e-3)
    optimizer_cbam = AdamW(cbam_params, lr=CONFIG["lr"], weight_decay=1e-3)

    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    scheduler_cbam = CosineAnnealingLR(optimizer_cbam, T_max=CONFIG["epochs"])
    # scheduler = OneCycleLR(
    #     optimizer,
    #     max_lr=CONFIG["lr"],
    #     total_steps=total_steps,
    #     pct_start=0.3,             # 30% of steps warming up
    #     anneal_strategy='cos',     # cosine annealing down
    #     div_factor=25.0,           # initial_lr = max_lr/div_factor
    #     final_div_factor=1e4       # min_lr = initial_lr/final_div_factor
    # )

    # scheduler_cbam = OneCycleLR(
    #     optimizer_cbam,
    #     max_lr=CONFIG["lr-cbam"],
    #     total_steps=total_steps,
    #     pct_start=0.3,
    #     anneal_strategy='cos',
    #     div_factor=25.0,
    #     final_div_factor=1e4
    # )

    scaler = GradScaler()

    early_stopping = EarlyStopping(
        patience=CONFIG["patience"],
        min_delta=CONFIG["min_delta"],
        monitor=CONFIG["monitor"]
    )

    print("starting loop...")
    for epoch in range(1, CONFIG["epochs"] + 1):
        # Reduce CBAM+SA learning rate at epoch 5
        if epoch == 5:
            for param_group in optimizer_cbam.param_groups:
                param_group['lr'] *= 0.5
            print(
                f"Reduced CBAM learning rate to {optimizer_cbam.param_groups[0]['lr']:.6f}")
        model.train()
        train_loss = 0.0
        train_correct = train_total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            optimizer_cbam.zero_grad()

            with autocast('cuda', dtype=torch.bfloat16):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
                error_if_nonfinite=False
            )
            scaler.step(optimizer)
            scaler.step(optimizer_cbam)
            scaler.update()

            preds = outputs.argmax(dim=1)
            train_loss += loss.item() * imgs.size(0)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total
        
        scheduler.step()
        scheduler_cbam.step()

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
            "val/f1": f1
        })

        print(f"Epoch {epoch:02d} | "
              f"Train loss {train_loss:.4f}, acc {train_acc:.4f} | "
              f"Val loss {val_loss:.4f}, acc {val_acc:.4f}")

        metrics = {"val_loss": val_loss, "val_acc": val_acc, "val_f1": f1}
        if early_stopping(model, metrics):
            print(f"Early stopping triggered after {epoch} epochs")
            break

    early_stopping.load_best_model(model)

    f1_history_np = np.stack(f1_history)
    plot_f1_history(f1_history_np, class_names)

    torch.save(model.state_dict(), CONFIG["save_path"])
    print(f"Best model saved to {CONFIG['save_path']}")


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


def visualize_shhit(loader, class_names):
    # Pobierz jedną paczkę
    imgs, labels = next(iter(loader))

    # imgs: tensor [B, C, H, W], labels: tensor [B]
    # Wybierz pierwsze 9 obrazów
    imgs = imgs[:9]
    labels = labels[:9]

    # Konwertuj na numpy i wstaw do siatki 3×3
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        img = imgs[i].cpu().permute(1, 2, 0).numpy()  # H×W×C
        # jeśli grayscale, to squeeze i cmap='gray'
        if img.shape[2] == 1:
            img = img.squeeze(-1)
            ax.imshow(img, cmap='gray')
        else:
            # zdjęcia po Normalize mają wartości w ~[-1,1], przeskaluj do [0,1]
            img = (img - img.min()) / (img.max() - img.min())
            ax.imshow(img)
        ax.set_title(class_names[labels[i].item()])
        ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
