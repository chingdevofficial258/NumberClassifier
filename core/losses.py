import numpy as np

class LossFunction:

    @staticmethod
    def cross_entropy(y_pred, y_true, eps: int = 1e-12):
        # Number of training samples
        m = len(y_true)

        # Probability assigned to the correct class
        correct_probs = y_pred[np.arange(m), y_true]

        # Calculate categorical cross-entropy loss
        return -np.mean(np.log(np.clip(correct_probs, eps, 1.0)))