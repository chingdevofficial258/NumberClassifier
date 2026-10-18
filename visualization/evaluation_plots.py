
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)


class EvaluationPlots:

    @staticmethod
    def classification_report(y_true, y_pred, save_dir="plots"):
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        report = classification_report(
            y_true,
            y_pred,
            zero_division=0
        )

        accuracy = accuracy_score(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.axis("off")

        ax.text(
            0.5, 0.95,
            f"Classification Report\nAccuracy: {accuracy:.2%}",
            ha="center",
            va="top",
            fontsize=15,
            fontweight="bold",
            transform=ax.transAxes
        )

        ax.text(
            0.5, 0.78,
            report,
            ha="center",
            va="top",
            fontfamily="monospace",
            fontsize=10,
            transform=ax.transAxes
        )

        fig.savefig(
            Path(save_dir) / "classification_report.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close(fig)

    @staticmethod
    def confusion_matrix(y_true, y_pred, save_dir="plots"):
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 7))

        ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            labels=list(range(10)),
            cmap="Blues",
            ax=ax
        )

        ax.set_title("Digit Classification Confusion Matrix")

        fig.savefig(
            Path(save_dir) / "confusion_matrix.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close(fig)
