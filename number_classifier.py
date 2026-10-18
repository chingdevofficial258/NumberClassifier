import numpy as np

from core import ActivationFunction, LossFunction

class NumberClassifier(ActivationFunction, LossFunction):
    def __init__(self,
                 input_size: int,
                 hidden_size1: int,
                 hidden_size2: int,
                 output_size: int,
                 learning_rate: int = 0.1):
        
        self.lr = learning_rate
        
        self.rng = np.random.default_rng(42)

        # Initialize param
        self.W1 = self.rng.uniform(-0.5, 0.5, (input_size, hidden_size1))
        self.b1 = np.zeros((1, 32))
        
        self.W2 = self.rng.uniform(-0.5, 0.5, (hidden_size1, hidden_size2))
        self.b2 = np.zeros((1, 20))
        
        self.W3 = self.rng.uniform(-0.5, 0.5, (hidden_size2, output_size))
        self.b3 = np.zeros((1, 10))

    def forward(self, X, y_true):
        Z1 = X @ self.W1 + self.b1 # calculate the dot production + bias 1
        self.A1 = self.sigmoid(Z1) # activation 1
    
        Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.sigmoid(Z2)
    
        Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = self.softmax(Z3)
    
        # Binary cross-entropy loss
        loss = self.cross_entropy(self.A3, y_true)
    
        # Calculate accuracy
        predictions = np.argmax(self.A3, axis=1)
        accuracy = np.mean(predictions == y_true)
    
        return (loss, accuracy,)

    def backward(self, X, y_true): # takes an activation function A1 and A2

        # Number of samples in the current batch
        m = len(y_true)
    
        # -------------------------
        # OUTPUT LAYER (W3)
        # -------------------------
        
        # Softmax + cross-entropy derivative
        dZ3 = self.A3.copy()
        dZ3[np.arange(m), y_true] -= 1
        dZ3 /= m

        # Calculate gradients for W3 and b3
        dW3 = self.A2.T @ dZ3
        db3 = np.sum(dZ3, axis=0, keepdims=True)
        
        # -------------------------
        # HIDDEN LAYER 2 (W2)
        # -------------------------
        
        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * self.A2 * (1 - self.A2)
    
        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)
    
        # -------------------------
        # HIDDEN LAYER 1 (W1)
        # -------------------------
        
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.A1 * (1 - self.A1)
    
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # -------------------------
        # UPDATE PARAMETERS
        # -------------------------

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3

    def predict(self, X, y):
        # Forward propagation
        loss, accuracy = self.forward(X, y)

        # Get predicted digits for all samples
        predictions = np.argmax(self.A3, axis=1)

        # Get confidence for each prediction
        confidence = self.A3[
            np.arange(len(predictions)),
            predictions
        ] * 100

        return loss, accuracy, predictions, confidence