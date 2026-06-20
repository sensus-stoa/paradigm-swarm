#!/usr/bin/env python3
"""Paradigm Swarm — Distortion Accumulation (TRUE sequential)
Monolithic: trained on P1, then P2 only, then P3 only — NO old data.
Paradigm Swarm: new Gaussian expert, old ones FROZEN.
"""
import numpy as np

np.random.seed(42)

N_train, N_test = 200, 100
centers = [np.array([-5,-5]), np.array([5,-5]), np.array([0,6]), np.array([-4,4]), np.array([4,4])]

# Generate
X_train_p, y_train_p = [], []
X_test_p, y_test_p = [], []
for i, c in enumerate(centers):
    X_train_p.append(np.random.randn(N_train,2)*1.2 + c)
    y_train_p.append(np.full(N_train, i))
    X_test_p.append(np.random.randn(N_test,2)*1.2 + c)
    y_test_p.append(np.full(N_test, i))

# ═══════════════════════════════════════════════════
# MONOLITHIC: True sequential (no old data revisited)
# ═══════════════════════════════════════════════════
class SeqMLP:
    def __init__(self):
        rng = np.random.RandomState(42)
        self.W1 = rng.randn(2, 20)*0.5; self.b1 = np.zeros((1,20))
        self.W2 = rng.randn(20, 5)*0.5; self.b2 = np.zeros((1,5))
    def forward(self, X):
        a1 = np.tanh(X @ self.W1 + self.b1)
        z = a1 @ self.W2 + self.b2
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
    def predict(self, X):
        return self.forward(X).argmax(axis=1)
    def train_on(self, X, class_idx, epochs=400, lr=0.05):
        """Train on ONE class only — no access to other classes."""
        # We need to train the model to recognize this class AND not forget others.
        # For forgetting to happen: train ONLY on this class with its label.
        # The model will push this class up and suppress others.
        y_local = np.full(X.shape[0], class_idx)
        y_oh = np.eye(5)[y_local]
        for _ in range(epochs):
            p = self.forward(X); N = X.shape[0]
            dz = (p - y_oh) / N
            a1 = np.tanh(X @ self.W1 + self.b1)
            self.W2 -= lr * a1.T @ dz; self.b2 -= lr * dz.sum(0, keepdims=True)
            da = dz @ self.W2.T
            self.W1 -= lr * X.T @ (da * (1-a1**2))
            self.b1 -= lr * (da * (1-a1**2)).sum(0, keepdims=True)

mono = SeqMLP()
mono_results = []

for p_idx in range(5):
    # Train ONLY on current paradigm
    mono.train_on(X_train_p[p_idx], p_idx)

    # Test on ALL paradigms seen so far
    accs = []
    for test_p in range(p_idx + 1):
        preds = mono.predict(X_test_p[test_p])
        acc = np.mean(preds == test_p)
        accs.append(acc)
    mono_results.append(accs)

# ═══════════════════════════════════════════════════
# PARADIGM SWARM: Gaussian experts, frozen
# ═══════════════════════════════════════════════════
class GaussExpert:
    def __init__(self, X):
        self.mean = X.mean(axis=0)
        self.cov = np.cov(X.T) + np.eye(2)*0.01
        self.inv_cov = np.linalg.inv(self.cov)
        self.det = np.linalg.det(self.cov)
    def density(self, X):
        diff = X - self.mean
        mahal = np.sum(diff @ self.inv_cov * diff, axis=1)
        return np.exp(-0.5 * mahal) / np.sqrt((2*np.pi)**2 * self.det)

ps_experts = []
ps_results = []

for p_idx in range(5):
    exp = GaussExpert(X_train_p[p_idx])
    ps_experts.append(exp)

    accs = []
    for test_p in range(p_idx + 1):
        X_t = X_test_p[test_p]
        densities = np.column_stack([e.density(X_t) for e in ps_experts])
        preds = densities.argmax(axis=1)
        acc = np.mean(preds == test_p)
        accs.append(acc)
    ps_results.append(accs)

# ═══════════════════════════════════════════════════
# PRINT
# ═══════════════════════════════════════════════════
print("="*70)
print("DISTORTION ACCUMULATION — True Sequential (no old data)")
print("="*70)

for step in range(5):
    print(f"\nStep {step+1}: trained on P{step+1} only")
    header = "        " + "  ".join(f"P{i+1}" for i in range(step+1))
    print(f"  {header}")
    m_str = "  ".join(f"{v:.3f}" for v in mono_results[step])
    p_str = "  ".join(f"{v:.3f}" for v in ps_results[step])
    print(f"  Mono:  {m_str}")
    print(f"  PS:    {p_str}")

    if step > 0:
        # Did P1 degrade?
        m_forget = mono_results[0][0] - mono_results[step][0] if len(mono_results[step]) > 0 else 0
        p_forget = ps_results[0][0] - ps_results[step][0] if len(ps_results[step]) > 0 else 0
        if m_forget > 0.05:
            print(f"  ⚠ Mono P1 forgot {m_forget:.3f}  |  PS P1 forgot {p_forget:.3f}")

# ═══════════════════════════════════════════════════
# FORGETTING CURVE
# ═══════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"P1 FORGETTING CURVE")
print(f"{'='*70}")
print(f"{'Step':<15} {'Mono P1 acc':>12} {'PS P1 acc':>12} {'Mono forget':>12}")
print("-"*55)
for step in range(5):
    m = mono_results[step][0] if step < len(mono_results) else 0
    p = ps_results[step][0] if step < len(ps_results) else 0
    forget = mono_results[0][0] - m if step > 0 else 0
    print(f"After P{step+1:<10} {m:>12.3f} {p:>12.3f} {forget:>+12.3f}")

final_m = mono_results[-1]
final_p = ps_results[-1]
print(f"\nFinal avg (5 paradigms): Mono={np.mean(final_m):.3f}  PS={np.mean(final_p):.3f}")
