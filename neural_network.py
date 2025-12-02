# neural_network.py

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def drelu(x):
    return (x > 0).astype(float)


class SimpleNN:
    """
    AlphaZero-style network:
      - Input: 84-dim vector (board_to_vector)
      - Hidden: ReLU
      - Outputs:
          * policy: length-7 softmax over moves
          * value: scalar in [0,1] (win probability for current player)
    """

    def __init__(self, hidden=128):
        self.input_size = 84
        self.hidden_size = hidden

        # Shared trunk
        self.W1 = np.random.uniform(-0.1, 0.1, (hidden, self.input_size))
        self.b1 = np.zeros(hidden)

        # Value head
        self.Wv = np.random.uniform(-0.1, 0.1, hidden)
        self.bv = 0.0

        # Policy head (7 moves)
        self.Wp = np.random.uniform(-0.1, 0.1, (7, hidden))
        self.bp = np.zeros(7)

    def _softmax(self, z):
        z = np.asarray(z, dtype=float)
        z_max = np.max(z)
        exps = np.exp(z - z_max)
        s = np.sum(exps)
        if s <= 0.0 or not np.isfinite(s):
            # fallback uniform
            return np.ones_like(z) / len(z)
        return exps / s

    def forward_policy_value(self, x):
        """
        Forward pass that returns (policy_probs, value_prob).

        x: 1D array-like of length 84.
        returns:
            policy: shape (7,) summing to 1
            value: scalar in [0,1]
        """
        x = np.asarray(x, dtype=float)
        self.x = x

        # Shared trunk
        self.z1 = self.W1 @ x + self.b1        # (hidden,)
        self.a1 = np.maximum(self.z1, 0.0)     # ReLU

        # Value head
        self.zv = self.Wv @ self.a1 + self.bv  # scalar
        self.value = sigmoid(self.zv)          # [0,1]

        # Policy head
        self.zp = self.Wp @ self.a1 + self.bp  # (7,)
        self.policy = self._softmax(self.zp)

        return self.policy, self.value

    def forward(self, x):
        """
        Backwards-compatible: returns only the value (scalar in [0,1]).
        Other code that expects model.forward(x) still works.
        """
        _, v = self.forward_policy_value(x)
        return v

    def backward_az(self, y_value, pi_target, lr=0.01,
                    value_weight=1.0, policy_weight=1.0):
        """
        Backprop for AlphaZero-style loss on a *single* example:

          L = value_weight * BCE(y_value, value)
              + policy_weight * CE(pi_target, policy)

        y_value: scalar in [0,1] (e.g., 1 for win, 0 for loss, 0.5 for draw)
        pi_target: length-7 target policy (probabilities, sum ~ 1).
        """
        pi_target = np.asarray(pi_target, dtype=float)

        # --- Value loss: BCE ---
        v = float(self.value)
        eps = 1e-7
        v_clamped = max(min(v, 1.0 - eps), eps)
        # dL_v/dzv = v - y  (for sigmoid + BCE)
        dzv = value_weight * (v - y_value)

        # Gradients for value head
        dWv = dzv * self.a1               # (hidden,)
        dbv = dzv

        # --- Policy loss: CE(pi_target, policy) ---
        p = np.asarray(self.policy, dtype=float)
        # dL_p/dzp = p - pi_target  (softmax + CE)
        dzp = policy_weight * (p - pi_target)  # (7,)

        # Gradients for policy head
        dWp = dzp[:, None] @ self.a1[None, :]  # (7, hidden)
        dbp = dzp

        # --- Backprop to shared hidden layer ---
        # from value head
        da1_v = self.Wv * dzv        # (hidden,)
        # from policy head
        da1_p = self.Wp.T @ dzp      # (hidden,)
        da1 = da1_v + da1_p          # combine

        dz1 = da1 * drelu(self.z1)   # (hidden,)

        dW1 = dz1[:, None] @ self.x[None, :]   # (hidden, 84)
        db1 = dz1

        # --- Parameter update ---
        self.Wv -= lr * dWv
        self.bv -= lr * dbv

        self.Wp -= lr * dWp
        self.bp -= lr * dbp

        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    # --- Save / load / clone (updated for new params) ---

    def save(self, path):
        np.savez(
            path,
            W1=self.W1,
            b1=self.b1,
            Wv=self.Wv,
            bv=self.bv,
            Wp=self.Wp,
            bp=self.bp,
        )

    @staticmethod
    def load(path):
        data = np.load(path)
        W1 = data["W1"]
        b1 = data["b1"]
        Wv = data["Wv"]
        bv = data["bv"]
        Wp = data["Wp"]
        bp = data["bp"]

        hidden = W1.shape[0]
        model = SimpleNN(hidden=hidden)
        model.W1 = W1
        model.b1 = b1
        model.Wv = Wv
        model.bv = bv
        model.Wp = Wp
        model.bp = bp
        return model

    def clone(self):
        new = SimpleNN(hidden=self.W1.shape[0])
        new.W1 = self.W1.copy()
        new.b1 = self.b1.copy()
        new.Wv = self.Wv.copy()
        new.bv = self.bv
        new.Wp = self.Wp.copy()
        new.bp = self.bp.copy()
        return new
