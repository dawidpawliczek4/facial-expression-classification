
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import wandb

def plot_confusion_matrix(cm, epoch, class_names=None):
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion Matrix (Epoch {epoch})")
    plt.tight_layout()

    wandb.log({f"confusion_matrix/epoch_{epoch}": wandb.Image(plt)})
    plt.close()


def plot_f1_history(f1_history, class_names=None):
    epochs = np.arange(1, f1_history.shape[0] + 1)
    plt.figure(figsize=(10, 6))
    if f1_history.ndim == 1:
        # Jedna krzywa (macro-F1)
        plt.plot(epochs, f1_history, marker='o', label='Macro F1')
    else:
        # Per-class
        for class_idx in range(f1_history.shape[1]):
            plt.plot(
                epochs,
                f1_history[:, class_idx],
                marker='o',
                label=(class_names[class_idx] if class_names else f"Class {class_idx}")
            )
    plt.xlabel("Epoch")
    plt.ylabel("F1-score")
    plt.title("Per-class F1-score over epochs")
    plt.legend()
    plt.tight_layout()

    wandb.log({"f1_history": wandb.Image(plt)})
    plt.close()
