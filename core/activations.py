import numpy as np

class ActivationFunction:
    
    # sigmoid hidden activation function
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    # softmax output activation function
    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)