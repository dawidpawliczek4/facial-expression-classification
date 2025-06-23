import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
from facexpr.utils.visualization import plot_confusion_matrix, plot_f1_history
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from facexpr.resnet18.model import ResNetClassifier
from facexpr.data.load_data import make_dataloaders

CONFIG = {
    "data_dir": "./data/downloaded_data/data",
    "batch_size": 64,
    "img_size": 224,
}


def main():
    wandb.init(project="fet2013-resnet", name="test-effnet-v1",
               config={"name": "test-resnet", "project": "fet2013-resnet"})
    class_names = ["Angry", "Disgust", "Fear",
                   "Happy", "Sad", "Surprise", "Neutral"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNetClassifier()
    checkpoint = torch.load('outputs/model.pth', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    loaders = make_dataloaders(        data_dir=CONFIG["data_dir"],
        batch_size=CONFIG["batch_size"],
        img_size=CONFIG['img_size'],
        num_workers=2,
        augment=False,
        grayscale=False)
    test_loader = loaders['test']

    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    explained_one = False
    with torch.no_grad():
        for imgs, labels in test_loader:            

            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    report = classification_report(y_true, y_pred, digits=3, zero_division=0)
    plot_confusion_matrix(cm, 0)
    plot_f1_history(np.stack([f1]), class_names)
    acc = correct / total

    wandb.log({
        f"classification_report/test": wandb.Table(
            data=[[line] for line in report.split("\n")], columns=["report"])})

    wandb.log({
        "test/accuracy": acc,
        "test/f1": f1,
    })


if __name__ == "__main__":
    main()
