import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def drelu(x):
    return (x > 0).astype(float)


class SimpleNN:
    def __init__(self, hidden = 64):
        self.input_size = 84
        self.W1 = np.random.uniform(-0.1, 0.1, (hidden, self.input_size))
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.uniform(-0.1, 0.1, hidden)
        self.b2 = 0.0

    def forward(self, x):
        self.x = x
        self.z1 = self.W1 @ x + self.b1          # shape (hidden,)
        self.a1 = np.maximum(self.z1, 0)         # ReLU
        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2
    
    def backward(self, y, lr=0.01):
        # Output layer
        dz2 = self.a2 - y                        # scalar
        dW2 = dz2 * self.a1                      # (hidden,)
        db2 = dz2

        # Hidden layer
        dz1 = (self.W2 * dz2) * drelu(self.z1)   # (hidden,)
        dW1 = dz1[:, None] @ self.x[None, :]     # (hidden,42)
        db1 = dz1

        # Update
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def save(self, path):
        """
        Save model weights to an .npz file.
        """
        np.savez(
            path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
        )

    @staticmethod
    def load(path):
        """
        Load model weights from an .npz file and return a new SimpleNN.
        """
        data = np.load(path)
        W1 = data["W1"]
        b1 = data["b1"]
        W2 = data["W2"]
        b2 = data["b2"]

        hidden = W1.shape[0]
        model = SimpleNN(hidden=hidden)
        model.W1 = W1
        model.b1 = b1
        model.W2 = W2
        model.b2 = b2
        return model

    def clone(self):
        new = SimpleNN(hidden=self.W1.shape[0])
        new.W1 = self.W1.copy()
        new.b1 = self.b1.copy()
        new.W2 = self.W2.copy()
        new.b2 = self.b2
        return new
model = SimpleNN()