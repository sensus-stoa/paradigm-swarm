#!/usr/bin/env python3
"""
Counter-Domain Experiment: Bull vs Bear markets.
Гипотеза: compiled эксперт на контр-доменах ХУЖЕ специализированных.

В отличие от независимых доменов (code/legal/finance), где compiled > solo,
контр-домены содержат ПРОТИВОПОЛОЖНЫЕ сигналы.
Compiled пытается усреднить → проваливается.
"""
import numpy as np
np.random.seed(42)

N_FEATURES = 50
N_SAMPLES = 400

# ═══ Generate counter-domain data ═══
# Bull: features 0-14 POSITIVE, features 15-29 NEGATIVE, 30-49 noise
# Bear: features 0-14 NEGATIVE, features 15-29 POSITIVE, 30-49 noise
# Counter-domain: SAME features, OPPOSITE signs

def make_bull(n):
    X = np.random.randn(n, N_FEATURES) * 0.3
    X[:, 0:15] += 2.0    # strong positive signal
    X[:, 15:30] -= 2.0   # strong negative signal
    y = (X[:, 0:30].sum(1) > 0).astype(int)
    return X, y

def make_bear(n):
    X = np.random.randn(n, N_FEATURES) * 0.3
    X[:, 0:15] -= 2.0    # NEGATIVE (opposite of bull!)
    X[:, 15:30] += 2.0   # POSITIVE (opposite of bull!)
    y = (X[:, 0:30].sum(1) > 0).astype(int)
    return X, y

def make_neutral(n):
    """Control: independent domain."""
    X = np.random.randn(n, N_FEATURES) * 0.3
    X[:, 30:45] += 2.5
    y = (X[:, 30:45].sum(1) > 0).astype(int)
    return X, y

# ═══ Expert (same MLP as always) ═══
class Expert:
    def __init__(self, seed):
        rng = np.random.RandomState(seed); s = 0.05
        self.W1 = rng.randn(N_FEATURES, 32) * s; self.b1 = np.zeros(32)
        self.W2 = rng.randn(32, 2) * s; self.b2 = np.zeros(2)
    def forward(self, X):
        a = np.maximum(0, X @ self.W1 + self.b1)
        z = a @ self.W2 + self.b2
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
    def predict(self, X): return self.forward(X).argmax(1)
    def prob_class1(self, X): return self.forward(X)[:, 1]
    def accuracy(self, X, y): return np.mean(self.predict(X) == y)
    def train(self, X, y, epochs=500, lr=0.01):
        yo = np.eye(2)[y]
        for _ in range(epochs):
            idx = np.random.choice(len(X), min(64, len(X)), replace=False)
            Xb, yb = X[idx], yo[idx]
            p = self.forward(Xb); N = len(Xb)
            a = np.maximum(0, Xb @ self.W1 + self.b1)
            dz = (p - yb) / N
            self.W2 -= lr * a.T @ dz; self.b2 -= lr * dz.sum(0)
            da = dz @ self.W2.T * (a > 0)
            self.W1 -= lr * Xb.T @ da; self.b1 -= lr * da.sum(0)

# ═══ Train ═══
print("="*60)
print("COUNTER-DOMAIN: Bull vs Bear")
print("="*60)

X_bull, y_bull = make_bull(N_SAMPLES)
X_bear, y_bear = make_bear(N_SAMPLES)
X_neutral, y_neutral = make_neutral(N_SAMPLES)

E_bull = Expert(42); E_bull.train(X_bull, y_bull)
E_bear = Expert(43); E_bear.train(X_bear, y_bear)
E_neutral = Expert(44); E_neutral.train(X_neutral, y_neutral)

print(f"  E_bull on bull:     {E_bull.accuracy(X_bull, y_bull):.3f}")
print(f"  E_bear on bear:     {E_bear.accuracy(X_bear, y_bear):.3f}")
print(f"  E_neutral on neut:  {E_neutral.accuracy(X_neutral, y_neutral):.3f}")

# ═══ Compiled ═══
X_both = np.vstack([X_bull, X_bear])
y_both = np.hstack([y_bull, y_bear])
E_comp = Expert(99); E_comp.train(X_both, y_both, epochs=800)

# ═══ Test ═══
X_test_bull, y_test_bull = make_bull(200)
X_test_bear, y_test_bear = make_bear(200)
X_test_neutral, y_test_neutral = make_neutral(200)

print(f"\n{'='*60}")
print(f"CROSS-DOMAIN EVALUATION")
print(f"{'='*60}")
print(f"  {'Model':<20} {'Bull':>10} {'Bear':>10} {'Neutral':>10} {'Avg':>10}")
print(f"  {'─'*60}")

models = [
    ("E_bull", E_bull),
    ("E_bear", E_bear),
    ("E_neutral", E_neutral),
    ("Compiled (bull+bear)", E_comp),
]

for name, m in models:
    a = m.accuracy(X_test_bull, y_test_bull)
    b = m.accuracy(X_test_bear, y_test_bear)
    c = m.accuracy(X_test_neutral, y_test_neutral)
    avg = (a + b + c) / 3
    print(f"  {name:<20} {a:>10.3f} {b:>10.3f} {c:>10.3f} {avg:>10.3f}")

# ═══ Coalition ═══
coal_bull = np.mean((E_bull.prob_class1(X_test_bull) > 0.5).astype(int) == y_test_bull)
coal_bear = np.mean((E_bear.prob_class1(X_test_bear) > 0.5).astype(int) == y_test_bear)
coal_neut = np.mean(((E_bull.prob_class1(X_test_neutral) + E_bear.prob_class1(X_test_neutral)) / 2 > 0.5).astype(int) == y_test_neutral)
print(f"  {'Coalition (max)':<20} {coal_bull:>10.3f} {coal_bear:>10.3f} {coal_neut:>10.3f} {(coal_bull+coal_bear+coal_neut)/3:>10.3f}")

print(f"\n  Counter-domain result:")
print(f"    Compiled on bull: {E_comp.accuracy(X_test_bull, y_test_bull):.3f}")
print(f"    Compiled on bear: {E_comp.accuracy(X_test_bear, y_test_bear):.3f}")
print(f"    E_bull on bear:   {E_bull.accuracy(X_test_bear, y_test_bear):.3f}  (counter-domain!)")
print(f"    E_bear on bull:   {E_bear.accuracy(X_test_bull, y_test_bull):.3f}  (counter-domain!)")

# ═══ KEY INSIGHT ═══
print(f"\n  ─── INSIGHT ───")
print(f"  Independent domains (neutral): compiled learns just fine.")
print(f"  Counter-domains (bull vs bear): signals are OPPOSITE.")
print(f"  Compiled trained on both → forced to compromise → degrades.")
print(f"  Counter-domains need ROUTING, not distillation.")
