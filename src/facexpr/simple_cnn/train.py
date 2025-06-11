import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from facexpr.data.load_data import make_dataloaders
from facexpr.models.cnn import SimpleCnnModel

def train():
    # TODO: train better CNN, cuz that's shit.
    """
    Trains a baseline CNN model for facial expression classification.

    Command-line Arguments:
        --data-dir (str): Path to the dataset directory (required).
        --batch-size (int): Batch size for training (default: 32).
        --epochs (int): Number of training epochs (default: 10).
        --lr (float): Learning rate for the optimizer (default: 1e-3).
        --img-size (int): Size to which input images are resized (default: 48).
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=48)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = make_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        augment=True
    )
    train_loader, val_loader = loaders["train"], loaders["val"]

    model = SimpleCnnModel().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs+1):
        model.train()
        running_loss = 0.0
        correct = total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
    
        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch:02d} | "
              f"Train loss {train_loss:.4f}, acc {train_acc:.4f} | "
              f"Val loss {val_loss:.4f}, acc {val_acc:.4f}")

    torch.save(model.state_dict(), "./outputs/models/baseline_cnn.pth")

if __name__ == "__main__":
    train()
