# NumberClassifier

**NumberClassifier** is a neural network built from scratch using Python and NumPy to classify handwritten digits from 0 to 9.

The project focuses on understanding the fundamental concepts of deep learning by implementing neural network operations manually, without relying on high-level machine learning frameworks such as TensorFlow or PyTorch.

### Key Features

- **Feedforward Neural Network:** Implements multiple fully connected layers for digit classification.
- **Activation Functions:** Uses Sigmoid for hidden layers and Softmax for multiclass output predictions.
- **Backpropagation:** Calculates gradients using the chain rule to update weights and biases.
- **Mini-Batch Gradient Descent:** Trains the network using shuffled batches of samples.
- **Cross-Entropy Loss:** Measures classification error during training.
- **Training Visualization:** Tracks loss and accuracy across epochs.
- **Model Evaluation:** Tests classification performance on unseen handwritten digit images.

### Dataset

Uses the `load_digits()` dataset from scikit-learn, containing 1,797 grayscale images with a resolution of 8 × 8 pixels, representing handwritten digits from 0 to 9.

### Technologies

Python · NumPy · scikit-learn · Matplotlib · Jupyter Notebook · tqdm
