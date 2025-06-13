import torch
import torch.nn as nn
import numpy as np
import wandb
from facexpr.data.load_data import make_dataloaders
from facexpr.efficientnet.model import EfficientNetV2Classifier
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from facexpr.utils.visualization import plot_confusion_matrix, plot_f1_history
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from torch.amp import autocast, GradScaler
from torch.optim import AdamW, SGD
from facexpr.utils.early_stopping import EarlyStopping

CONFIG = {
    "data_dir": "./data/downloaded_data/data",
    "save_path": "./outputs/models/model.pth",
    "img_size": 224,
    "project": "fer2013-efficientnetv2",
    "name": "16-cbam-multiheadatt",    
    # Early stopping parameters
    "patience": 5,
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


def make_optimizer(params, opt, lr, weight_decay):
    if opt == "AdamW":
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        return SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)

def make_scheduler(optim, sched, epochs, step_size, gamma):
    if sched == "cosine":
        return CosineAnnealingLR(optim, T_max=epochs)
    else:
        return StepLR(optim, step_size=step_size, gamma=gamma)

def main():
    wandb.init(project=CONFIG["project"], config=CONFIG, name=CONFIG["name"])
    cfg = wandb.config

    class_names = ["Angry", "Disgust", "Fear",
                   "Happy", "Sad", "Surprise", "Neutral"]
    f1_history = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("loading data...")
    loaders = make_dataloaders(
        data_dir=CONFIG["data_dir"],
        batch_size=cfg.batch_size,
        img_size=CONFIG['img_size'],
        num_workers=2,
        augment=True,
        grayscale=False
    )
    train_loader, val_loader = loaders["train"], loaders["val"]

    model = EfficientNetV2Classifier(num_classes=7).to(device)

    def is_cbam_mhsa(name):
        res = any(tag in name.lower() for tag in ["cbam", "channel_att", "spatial_att", "self_attention", "mhsa", "attention", "att"])
        return res
    cbam_params = [p for n, p in model.named_parameters() if is_cbam_mhsa(n)]
    other_params = [p for n, p in model.named_parameters() if not is_cbam_mhsa(n)]

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    optimizer = make_optimizer(other_params, cfg.optimizer, cfg.lr, cfg.weight_decay)
    optimizer_cbam = make_optimizer(cbam_params, cfg.optimizer_cbam, cfg.lr_cbam, cfg.weight_decay)

    scheduler = make_scheduler(optimizer, cfg.scheduler, cfg.epochs, cfg.step_size, cfg.gamma)
    scheduler_cbam = make_scheduler(optimizer_cbam, cfg.scheduler_cbam, cfg.epochs, cfg.step_size, cfg.gamma)
    scaler = GradScaler()
    
    early_stopping = EarlyStopping(
        patience=CONFIG["patience"],
        min_delta=CONFIG["min_delta"],
        monitor=CONFIG["monitor"]
    )

    print("starting loop...")
    for epoch in range(1, cfg.epochs + 1):

        if epoch == 5 and cfg.reduce_lr_for_cbam:
            for param_group in optimizer_cbam.param_groups:
                param_group['lr'] *= 0.5
            print(f"Reduced CBAM learning rate to {optimizer_cbam.param_groups[0]['lr']:.6f}")

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

        scheduler.step()
        scheduler_cbam.step()

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


if __name__ == "__main__":
    main()