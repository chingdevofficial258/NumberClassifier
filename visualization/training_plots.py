
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

class TrainingPlots:

    @staticmethod
    def plot_loss(loss_history, save_dir="plots"):
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(9, 4))
        plt.plot(loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Sparse Categorical Cross-Entropy Loss")
        plt.title("Training Loss")
        plt.grid(True)

        plt.savefig(
            Path(save_dir) / "training_loss.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    @staticmethod
    def plot_accuracy(accuracy_history, save_dir="plots"):
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(9, 4))
        plt.plot(np.array(accuracy_history) * 100)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Training Accuracy")
        plt.ylim(0, 105)
        plt.grid(True)

        plt.savefig(
            Path(save_dir) / "training_accuracy.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()
