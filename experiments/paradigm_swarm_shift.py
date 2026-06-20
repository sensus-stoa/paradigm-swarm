#!/usr/bin/env python3
"""Paradigm Swarm — Boundary Shift Experiment
Shows: adding paradigm 4 shifts monolithic boundaries everywhere.
Paradigm Swarm experts stay frozen — zero shift.
"""
import numpy as np

np.random.seed(42)
N = 120

# 3 initial paradigms
centers_3 = {"A": np.array([-4, -4]), "B": np.array([4, -4]), "C": np.array([0, 5])}
X3, y3 = [], []
for i, c in enumerate(centers_3.values()):
    X3.append(np.random.randn(N, 2) * 1.0 + c)
    y3.append(np.full(N, i))
X3 = np.vstack(X3); y3 = np.hstack(y3)

# 4th paradigm (added later)
center_D = np.array([0, -4])
X_D = np.random.randn(N, 2) * 1.0 + center_D
y_D = np.full(N, 3)

# ─── Tiny MLPs ───
class MiniMLP:
    def __init__(self, n_out, seed=42):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(2, 12)*0.5; self.b1 = np.zeros((1,12))
        self.W2 = rng.randn(12, n_out)*0.5; self.b2 = np.zeros((1,n_out))
        self.n_out = n_out
    def forward(self, X):
        a1 = np.tanh(X @ self.W1 + self.b1)
        z2 = a1 @ self.W2 + self.b2
        if self.n_out == 1:
            return 1/(1+np.exp(-z2))
        e = np.exp(z2 - z2.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    def predict_class(self, X):
        probs = self.forward(X)
        if self.n_out == 1:
            return (probs[:,0] > 0.5).astype(int)
        return probs.argmax(axis=1)
    def train(self, X, y, epochs=400, lr=0.05):
        for _ in range(epochs):
            if self.n_out == 1:
                prob = self.forward(X)
                dz2 = (prob - y.reshape(-1,1)) / X.shape[0]
            else:
                probs = self.forward(X)
                y_oh = np.eye(self.n_out)[y]
                dz2 = (probs - y_oh) / X.shape[0]
            a1 = np.tanh(X @ self.W1 + self.b1)
            self.W2 -= lr * a1.T @ dz2; self.b2 -= lr * dz2.sum(0, keepdims=True)
            da1 = dz2 @ self.W2.T
            dz1 = da1 * (1 - a1**2)
            self.W1 -= lr * X.T @ dz1; self.b1 -= lr * dz1.sum(0, keepdims=True)

# ═══════════════════════════════════════════════════
# PHASE 1: Train on 3 paradigms
# ═══════════════════════════════════════════════════
print("="*65)
print("BOUNDARY SHIFT EXPERIMENT")
print("="*65)

# Monolithic (3-class)
mono3 = MiniMLP(3, seed=42)
mono3.train(X3, y3)

# Paradigm Swarm: 3 binary experts
ps3 = [MiniMLP(1, seed=42+i) for i in range(3)]
for i, exp in enumerate(ps3):
    exp.train(X3, (y3==i).astype(int))

# Save predictions on a test grid BEFORE adding paradigm D
grid_x = np.linspace(-8, 8, 60)
grid_y = np.linspace(-8, 8, 60)
xx, yy = np.meshgrid(grid_x, grid_y)
grid = np.c_[xx.ravel(), yy.ravel()]

mono_before = mono3.predict_class(grid)
ps_before = np.array([(np.column_stack([e.forward(grid)[:,0] for e in ps3]).argmax(axis=1))])

# Also save per-expert probabilities for boundary tracking
ps_probs_before = np.column_stack([e.forward(grid)[:,0] for e in ps3])

# ═══════════════════════════════════════════════════
# PHASE 2: Add paradigm D
# ═══════════════════════════════════════════════════
X_all = np.vstack([X3, X_D])
y_all = np.hstack([y3, y_D])

# Monolithic: retrain from scratch with 4 classes
mono4 = MiniMLP(4, seed=42)
mono4.train(X_all, y_all)

# Paradigm Swarm: add ONE new expert. Old 3 experts FROZEN.
ps4_experts = ps3 + [MiniMLP(1, seed=45)]  # old 3 + new 1
ps4_experts[3].train(X_all, (y_all==3).astype(int))  # only train new expert

# Predictions AFTER
mono_after = mono4.predict_class(grid)
ps_after_probs = np.column_stack([e.forward(grid)[:,0] for e in ps4_experts])
ps_after = ps_after_probs.argmax(axis=1)

# ═══════════════════════════════════════════════════
# MEASURE BOUNDARY SHIFT
# ═══════════════════════════════════════════════════
# How many grid points changed classification for classes A,B,C?
# (ignoring points that became class D — that's expected)
old_abc = (mono_before < 3)  # points classified as A,B,C before
new_abc = (mono_after < 3)   # points still A,B,C after
shifted_mono = (mono_before != mono_after) & old_abc & new_abc
shift_pct_mono = 100 * shifted_mono.sum() / old_abc.sum()

# Paradigm Swarm: old experts unchanged (frozen weights)
# So ps_before[0] for classes A,B,C should equal ps_after for same
ps_before_flat = ps_before.flatten()
ps_after_flat = ps_after.flatten()
old_abc_ps = (ps_before_flat < 3)
shifted_ps = (ps_before_flat != ps_after_flat) & old_abc_ps & (ps_after_flat < 3)
shift_pct_ps = 100 * shifted_ps.sum() / max(1, old_abc_ps.sum())

# Also measure: did old expert probabilities change?
expert_A_before = ps_probs_before[:,0]
expert_A_after = ps4_experts[0].forward(grid)[:,0]
expert_A_shift = np.abs(expert_A_after - expert_A_before).max()

print(f"\nGrid: {len(grid)} points")
print(f"\nMonolithic (3→4 classes):")
print(f"  Points that changed class: {shifted_mono.sum()}/{old_abc.sum()} ({shift_pct_mono:.1f}%)")
print(f"\nParadigm Swarm (frozen experts + new):")
print(f"  Points that changed class: {shifted_ps.sum()}/{old_abc_ps.sum()} ({shift_pct_ps:.1f}%)")
print(f"  Expert A probability max shift: {expert_A_shift:.6f} (should be 0)")
print(f"\nMonolithic: adding paradigm D reshuffled {shift_pct_mono:.0f}% of old decisions.")
print(f"Paradigm Swarm: old experts frozen → {shift_pct_ps:.0f}% shift (should be 0).")

if shift_pct_ps < 1 and expert_A_shift < 0.001:
    print(f"\n✓ STRUCTURAL PROOF: weight isolation prevents boundary shift.")
    print(f"  Old paradigms unaffected by new paradigm addition.")
else:
    print(f"\n⚠ Shift detected — check implementation.")
