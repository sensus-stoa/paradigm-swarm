#!/usr/bin/env python3
"""Paradigm Swarm: Weight Isolation Experiment — Pure NumPy
Proves: isolated weight training prevents catastrophic forgetting.
"""
import numpy as np
import time

print("="*65)
print("PARADIGM SWARM — Weight Isolation Experiment (NumPy)")
print("="*65)

# ─── Tiny Neural Network from scratch ─────────────
class TinyMLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01, seed=42):
        self.lr = lr
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = rng.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.probs = self._softmax(self.z2)
        return self.probs

    def _softmax(self, x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def train_step(self, X, y_onehot):
        probs = self.forward(X)
        N = X.shape[0]
        # Cross-entropy gradient
        dz2 = (probs - y_onehot) / N
        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (1 - np.tanh(self.z1)**2)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)
        # Update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def clone_weights(self):
        return {
            'W1': self.W1.copy(), 'b1': self.b1.copy(),
            'W2': self.W2.copy(), 'b2': self.b2.copy()
        }

    def restore_weights(self, weights):
        for k in ['W1','b1','W2','b2']:
            setattr(self, k, weights[k].copy())

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

# ─── Generate 2 synthetic tasks ───────────────────
# Task A: features 0-3 informative
# Task B: features 6-9 informative (DIFFERENT features)
np.random.seed(42)
N_train, N_test = 1000, 500
D = 10  # input dimension

def make_data(n, informative_start, flip=0.02):
    """Generate binary classification with informative features at given start."""
    X = np.random.randn(n, D) * 0.5
    # Label determined by sum of informative features
    y_logit = X[:, informative_start:informative_start+4].sum(axis=1)
    y = (y_logit > 0).astype(int)
    # Add noise
    flip_mask = np.random.rand(n) < flip
    y[flip_mask] = 1 - y[flip_mask]
    return X, y

X_A_train, y_A_train = make_data(N_train, 0)
X_A_test, y_A_test = make_data(N_test, 0)
X_B_train, y_B_train = make_data(N_train, 6)
X_B_test, y_B_test = make_data(N_test, 6)

def to_onehot(y, n_classes=2):
    oh = np.zeros((len(y), n_classes))
    oh[np.arange(len(y)), y] = 1
    return oh

y_A_train_oh = to_onehot(y_A_train)
y_B_train_oh = to_onehot(y_B_train)

# ─── CONDITION 1: Standard (NO isolation) ─────────
print("\n--- Condition 1: Standard (same network, sequential training) ---")
net_std = TinyMLP(D, 32, 2, lr=0.05, seed=42)

# Train on A
for epoch in range(200):
    net_std.train_step(X_A_train, y_A_train_oh)
acc_A_after_A = net_std.score(X_A_test, y_A_test)
print(f"After 200 epochs on A:         acc_A = {acc_A_after_A:.3f}")

# Save weights after A
weights_after_A = net_std.clone_weights()

# Train on B (same network — should cause forgetting)
for epoch in range(200):
    net_std.train_step(X_B_train, y_B_train_oh)
acc_B_std = net_std.score(X_B_test, y_B_test)
acc_A_after_B = net_std.score(X_A_test, y_A_test)
forgetting_std = acc_A_after_A - acc_A_after_B
print(f"After 200 epochs on B:         acc_B = {acc_B_std:.3f}")
print(f"After B:                       acc_A = {acc_A_after_B:.3f}")
print(f"FORGETTING:                    {forgetting_std:+.3f}")

# ─── CONDITION 2: Paradigm Swarm (ISOLATED) ───────
print("\n--- Condition 2: Paradigm Swarm (weight isolation) ---")
# Expert A
net_A = TinyMLP(D, 32, 2, lr=0.05, seed=42)
for epoch in range(200):
    net_A.train_step(X_A_train, y_A_train_oh)
acc_ps_A_init = net_A.score(X_A_test, y_A_test)
weights_A_frozen = net_A.clone_weights()
print(f"Expert A trained:              acc_A = {acc_ps_A_init:.3f}")

# Expert B — ISOLATED (separate network)
net_B = TinyMLP(D, 32, 2, lr=0.05, seed=99)
for epoch in range(200):
    net_B.train_step(X_B_train, y_B_train_oh)
acc_ps_B = net_B.score(X_B_test, y_B_test)
print(f"Expert B trained (isolated):   acc_B = {acc_ps_B:.3f}")

# Test Expert A after B training (should be UNCHANGED since isolated)
acc_ps_A_after_B = net_A.score(X_A_test, y_A_test)
forgetting_ps = acc_ps_A_init - acc_ps_A_after_B
print(f"Expert A after B (isolated):   acc_A = {acc_ps_A_after_B:.3f}")
print(f"FORGETTING:                    {forgetting_ps:+.3f}")

# ─── CONDITION 3: Router + Experts ────────────────
print("\n--- Condition 3: Paradigm Swarm + Router ---")
# Simple router: features 0-3 higher variance → task A, else B
def router(X):
    var_early = np.var(X[:, :4], axis=1).mean()
    var_late = np.var(X[:, 6:10], axis=1).mean()
    return "A" if var_early > var_late else "B"

# Test on combined data
X_combined = np.vstack([X_A_test, X_B_test])
y_combined = np.hstack([y_A_test, y_B_test])
true_task = np.hstack([np.zeros(N_test), np.ones(N_test)])  # 0=A, 1=B

# Router accuracy
route_pred = np.array([0 if router(X_combined[i:i+1]) == "A" else 1 for i in range(2*N_test)])
router_acc = np.mean(route_pred == true_task)
print(f"Router task-identification:    {router_acc:.3f}")

# Paradigm Swarm: route then classify
ps_preds = []
for i in range(2*N_test):
    r = router(X_combined[i:i+1])
    net = net_A if r == "A" else net_B
    ps_preds.append(net.predict(X_combined[i:i+1])[0])
ps_acc = np.mean(np.array(ps_preds) == y_combined)
print(f"Paradigm Swarm (route+experts): acc = {ps_acc:.3f}")

# ─── SUMMARY ──────────────────────────────────────
print("\n" + "="*65)
print("SUMMARY")
print("="*65)
print(f"{'Condition':<38} {'Acc A after B':>10} {'Forgetting':>12} {'Acc B':>10}")
print("-"*65)
print(f"{'Standard (same network)':<38} {acc_A_after_B:>10.3f} {forgetting_std:>+12.3f} {acc_B_std:>10.3f}")
print(f"{'Paradigm Swarm (isolated experts)':<38} {acc_ps_A_after_B:>10.3f} {forgetting_ps:>+12.3f} {acc_ps_B:>10.3f}")
print(f"{'Paradigm Swarm + Router':<38} {'N/A':>10} {'N/A':>12} {ps_acc:>10.3f}")

if forgetting_std > 0.01:
    reduction = 100 * (forgetting_std - forgetting_ps) / forgetting_std
    print(f"\nParadigm isolation PREVENTED {forgetting_std-forgetting_ps:.3f} points of forgetting.")
    print(f"Catastrophic forgetting reduced by: {reduction:.0f}%")
elif forgetting_ps < 0.001:
    print(f"\nZero forgetting with paradigm isolation — ARCHITECTURAL PROOF.")
