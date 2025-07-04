import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from facexpr.data.load_data import make_dataloaders
from facexpr.simple_cnn.resnet import ResNet50Classifier
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# Replace argparse with a simple CONFIG dictionary
CONFIG = {
    "data_dir": "/path/to/data/downloaded_data", # <-- set your path here
    "batch_size": 32,
    "epochs": 10,
    "lr": 1e-3,
    "save_path": "outputs/models/model.pth",
    "log_dir": "outputs/logs"
}

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
    f1 = f1_score(y_true, y_pred, average='macro')
    report = classification_report(y_true, y_pred, digits=3)
    return cm, f1, report, val_loss, val_correct, val_total

def plot_confusion_matrix(cm, epoch, log_dir, class_names=None):
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion Matrix (Epoch {epoch})")
    fname = os.path.join(log_dir, f"confusion_matrix_epoch_{epoch:02d}.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def plot_f1_history(f1_history, log_dir, class_names=None):
    epochs = np.arange(1, len(f1_history) + 1)
    plt.figure(figsize=(10, 6))
    for class_idx in range(f1_history.shape[1]):
        plt.plot(epochs, f1_history[:,class_idx], label=f"{class_names[class_idx] if class_names else 'Class '+str(class_idx)}")
    plt.xlabel("Epoch")
    plt.ylabel("F1-score")
    plt.title("Per-class F1-score over epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "f1_per_class_history.png"))
    plt.close()

def main():
    args = CONFIG

    os.makedirs(os.path.dirname(args["save_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(args["log_dir"]), exist_ok=True)

    class_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    f1_history = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Target size 224 for ResNet
    target_size = 224
    loaders = make_dataloaders(
        data_dir=args["data_dir"],
        batch_size=args["batch_size"],
        img_size=target_size,
        num_workers=2,
        augment=True,
        channels=3
    )
    train_loader, val_loader = loaders["train"], loaders["val"]

    model = ResNet50Classifier(num_classes=7, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=args["epochs"])

    # Freeze parameters of ResNet
    for param in model.backbone.parameters():
        param.requires_grad = False
    for name, param in model.backbone.named_parameters():
        if name.startswith("layer4") or name.startswith("fc"):
            param.requires_grad = True

    for epoch in range(1, args["epochs"] + 1):
        model.train()
        train_loss = 0.0
        train_correct = train_total = 0

        # Unfreeze some - "gradual unfreeze"
        if epoch == args["epochs"] // 2:
            for param in model.backbone.layer3.parameters():
                param.requires_grad = True

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()            

            train_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()

        train_loss /= train_total
        train_acc = train_correct / train_total

        cm, f1, report, val_loss, val_correct, val_total = evaluate_model(model, val_loader, device, criterion)

        f1_history.append(f1)
        plot_confusion_matrix(cm, epoch, args["log_dir"], class_names)

        val_loss /= val_total
        val_acc = val_correct / val_total

        all_report_path = os.path.join(args["log_dir"], "classification_reports.txt")
        with open(all_report_path, "a") as f:
            f.write(f"\n==== Epoch {epoch:02d} ====\n")
            f.write(report)

        print(f"Epoch {epoch:02d} | "
              f"Train loss {train_loss:.4f}, acc {train_acc:.4f} | "
              f"Val loss {val_loss:.4f}, acc {val_acc:.4f}")

    torch.save(model.state_dict(), args["save_path"])
    print(f"Model saved to {args['save_path']}")

    f1_history_np = np.stack(f1_history)
    plot_f1_history(f1_history_np, args["log_dir"], class_names)

if __name__ == "__main__":
    main()
