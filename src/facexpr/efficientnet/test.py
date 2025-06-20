import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
from facexpr.utils.visualization import plot_confusion_matrix, plot_f1_history
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from facexpr.efficientnet.model import EfficientNetV2Classifier
from facexpr.data.load_data import make_dataloaders


def main():
    class_names = ["Angry", "Disgust", "Fear",
                   "Happy", "Sad", "Surprise", "Neutral"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EfficientNetV2Classifier()
    checkpoint = torch.load('outputs/models/effnet.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    loaders = make_dataloaders()
    test_loader = loaders['test']

    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    explained_one = False
    with torch.no_grad():
        for imgs, labels in test_loader:
            if not explained_one:
                outputs, ch_map, sp_map, attn = model(imgs, explain=True)
                img = imgs[0].cpu().permute(1,2,0).numpy()
                sp = sp_map[0,0].cpu().numpy()
                ch = ch_map[0].squeeze().cpu().numpy()
                head0 = attn[0,0].cpu().numpy().reshape(sp.shape)
                fig, ax = plt.subplots(1,3, figsize=(12,4))
                #spatial
                ax[0].imshow(img)
                ax[0].imshow(sp, cmap='jet', alpha=0.5)
                ax[0].set_title("Spatial Map")
                ax[0].axis('off')
                #channel
                ax[1].bar(range(len(ch)), ch)
                ax[1].set_title('Channel Map')
                #mhsa
                ax[2].imshow(head0, cmap='hot')
                ax[2].set_title("MHSA Map")
                ax[2].axis('off')                
                wandb.log({"test/attention_expl": wandb.Image(fig)})
                plt.close(fig)
                explained_one = True

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
