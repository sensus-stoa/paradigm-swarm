#!/usr/bin/env python3
"""
Experiment: Density Restoration — Swarm as Teacher for Forgetting Transformer

Сценарий:
- Трансформер (MLP с общими весами) обучается последовательно на T0→T4.
  К T4 он забывает T0 (catastrophic forgetting).
- Paradigm Swarm (изолированные эксперты) обучается на тех же задачах.
  Эксперты не забывают — accuracy T0 стабильна.
- Density restoration: берём забытую задачу T0, используем эксперта T0
  как учителя (его predictions = pseudo-labels), дообучаем трансформер
  на T0-данных с этими метками.
- Измеряем: сколько accuracy возвращается.

Варианты восстановления:
  (a) Solo expert A → возвращает плотность задачи A в трансформер
  (b) Coalition A+B → возвращает кросс-доменную плотность
  (c) Compiled AB → возвращает joint-плотность
"""
import numpy as np

np.random.seed(42)
DIM, HIDDEN = 60, 64
N_TASKS = 5
N_TRAIN, N_TEST = 300, 150
EPOCHS = 600
LR = 0.01

# ─── Feature spaces (overlapping, like coalition experiments) ───
FEATURES = {
    0: list(range(0, 12)),                     # A
    1: list(range(8, 20)),                      # B (4 overlap with A)
    2: list(range(30, 42)),                     # C (isolated)
    3: list(range(16, 28)),                     # D (4 overlap with B)
    4: list(range(40, 52)),                     # E (some overlap with C)
}

def make_task(feats, n):
    X = np.random.randn(n, DIM) * 0.3
    X[:, feats] *= 6
    y = ((X[:, feats].sum(1)) > 0).astype(int)
    X += np.random.randn(*X.shape) * 0.5
    return X, y

# ─── Transformer (shared MLP, sequential training → forgetting) ───
class Transformer:
    def __init__(self):
        rng = np.random.RandomState(42); s = 0.05
        self.W1 = rng.randn(DIM, HIDDEN) * s
        self.b1 = np.zeros((1, HIDDEN))
        self.W2 = rng.randn(HIDDEN, 2) * s
        self.b2 = np.zeros((1, 2))

    def forward(self, X):
        a = np.maximum(0, X @ self.W1 + self.b1)
        z = a @ self.W2 + self.b2
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def predict(self, X): return self.forward(X).argmax(1)
    def prob_class1(self, X): return self.forward(X)[:, 1]
    def accuracy(self, X, y): return np.mean(self.predict(X) == y)

    def log_density(self, X):
        probs = self.forward(X)
        return np.log(probs.max(1) + 1e-8).mean()

    def train(self, X, y, epochs=EPOCHS, lr=LR):
        yo = np.eye(2)[y]
        for _ in range(epochs):
            idx = np.random.choice(len(X), min(64, len(X)), replace=False)
            Xb, yb = X[idx], yo[idx]
            p = self.forward(Xb); N = len(Xb)
            a = np.maximum(0, Xb @ self.W1 + self.b1)
            dz = (p - yb) / N
            self.W2 -= lr * a.T @ dz
            self.b2 -= lr * dz.sum(0, keepdims=True)
            da = dz @ self.W2.T * (a > 0)
            self.W1 -= lr * Xb.T @ da
            self.b1 -= lr * da.sum(0, keepdims=True)

    def train_with_teacher(self, X, teacher_probs, epochs=400, lr=LR):
        """Train using teacher's probability outputs as soft targets."""
        for _ in range(epochs):
            idx = np.random.choice(len(X), min(64, len(X)), replace=False)
            Xb = X[idx]
            p = self.forward(Xb); N = len(Xb)
            # KL-divergence style loss: push towards teacher's distribution
            t_probs = teacher_probs[idx]
            # Teacher gives class-1 prob → convert to 2-class distribution
            t_dist = np.column_stack([1 - t_probs, t_probs])
            a = np.maximum(0, Xb @ self.W1 + self.b1)
            dz = (p - t_dist) / N
            self.W2 -= lr * a.T @ dz
            self.b2 -= lr * dz.sum(0, keepdims=True)
            da = dz @ self.W2.T * (a > 0)
            self.W1 -= lr * Xb.T @ da
            self.b1 -= lr * da.sum(0, keepdims=True)

# ─── Expert (isolated, no forgetting) ───
class Expert:
    def __init__(self, seed):
        rng = np.random.RandomState(seed); s = 0.05
        self.W1 = rng.randn(DIM, HIDDEN) * s; self.b1 = np.zeros((1, HIDDEN))
        self.W2 = rng.randn(HIDDEN, 2) * s; self.b2 = np.zeros((1, 2))
    def forward(self, X):
        a = np.maximum(0, X @ self.W1 + self.b1)
        z = a @ self.W2 + self.b2
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
    def predict(self, X): return self.forward(X).argmax(1)
    def prob_class1(self, X): return self.forward(X)[:, 1]
    def accuracy(self, X, y): return np.mean(self.predict(X) == y)
    def train(self, X, y, epochs=EPOCHS, lr=LR):
        yo = np.eye(2)[y]
        for _ in range(epochs):
            idx = np.random.choice(len(X), min(64, len(X)), replace=False)
            Xb, yb = X[idx], yo[idx]
            p = self.forward(Xb); N = len(Xb)
            a = np.maximum(0, Xb @ self.W1 + self.b1)
            dz = (p - yb) / N
            self.W2 -= lr * a.T @ dz; self.b2 -= lr * dz.sum(0, keepdims=True)
            da = dz @ self.W2.T * (a > 0)
            self.W1 -= lr * Xb.T @ da; self.b1 -= lr * da.sum(0, keepdims=True)

# ─── Generate tasks ───
print("=" * 70)
print("DENSITY RESTORATION: Swarm → Transformer")
print("=" * 70)

X_train, y_train = {}, {}
X_test, y_test = {}, {}
for tid in range(N_TASKS):
    X_train[tid], y_train[tid] = make_task(FEATURES[tid], N_TRAIN)
    X_test[tid], y_test[tid] = make_task(FEATURES[tid], N_TEST)

# ─── Train experts (no forgetting) ───
experts = {}
for tid in range(N_TASKS):
    e = Expert(100 + tid)
    e.train(X_train[tid], y_train[tid])
    experts[tid] = e
    acc = e.accuracy(X_test[tid], y_test[tid])
    print(f"  Expert T{tid}: acc={acc:.3f}  (feats {FEATURES[tid][0]}..{FEATURES[tid][-1]})")

# ─── Compiled AB ───
X_ab_train = np.vstack([X_train[0], X_train[1]])
y_ab_train = np.hstack([y_train[0], y_train[1]])
compiled_AB = Expert(200)
compiled_AB.train(X_ab_train, y_ab_train, epochs=1200)

# ─── Train transformer sequentially → forgetting ───
print(f"\n{'─'*70}")
print(f"TRANSFORMER: Sequential training T0→T4")
print(f"{'─'*70}")

tf = Transformer()
tf_acc_history = {}

for tid in range(N_TASKS):
    tf.train(X_train[tid], y_train[tid])
    # Measure accuracy on ALL seen tasks after each new task
    accs = [tf.accuracy(X_test[t], y_test[t]) for t in range(tid + 1)]
    tf_acc_history[tid] = accs
    tf_density = tf.log_density(X_test[0]) if tid > 0 else 0
    print(f"  After T{tid}: T0 acc={accs[0]:.3f}  density(T0)={tf_density:.4f}  " +
          (f"avg={np.mean(accs):.3f}" if tid > 0 else ""))

tf_final_accs = [tf.accuracy(X_test[t], y_test[t]) for t in range(N_TASKS)]
tf_before_density = {t: tf.log_density(X_test[t]) for t in range(N_TASKS)}
print(f"\n  Final transformer (after T4): avg={np.mean(tf_final_accs):.3f}")
print(f"  T0 forgetting: {tf_acc_history[0][0]:.3f} → {tf_final_accs[0]:.3f}  Δ={tf_final_accs[0] - tf_acc_history[0][0]:+.3f}")

# ─── DENSITY RESTORATION ───
print(f"\n{'='*70}")
print(f"DENSITY RESTORATION: Expert → Teacher → Transformer Recovery")
print(f"{'='*70}")

# Strategy A: Restore T0 using Expert T0 as teacher
print(f"\n  ── Strategy A: Solo Expert T0 → restore T0 density ──")
tf_a = Transformer()
# First, train normally on T0→T4 (same forgetting baseline)
for tid in range(N_TASKS):
    tf_a.train(X_train[tid], y_train[tid])
acc_a_before = tf_a.accuracy(X_test[0], y_test[0])
density_a_before = tf_a.log_density(X_test[0])

# Restore: use Expert T0's predictions as teacher
teacher_probs = experts[0].prob_class1(X_train[0])
tf_a.train_with_teacher(X_train[0], teacher_probs, epochs=400)

acc_a_after = tf_a.accuracy(X_test[0], y_test[0])
density_a_after = tf_a.log_density(X_test[0])
acc_a_all_after = [tf_a.accuracy(X_test[t], y_test[t]) for t in range(N_TASKS)]

print(f"    T0 accuracy: {acc_a_before:.3f} → {acc_a_after:.3f}  Δ={acc_a_after - acc_a_before:+.3f}")
print(f"    T0 density:  {density_a_before:.4f} → {density_a_after:.4f}")
print(f"    Expert T0 reference accuracy: {experts[0].accuracy(X_test[0], y_test[0]):.3f}")
print(f"    All tasks after restore: {[f'{a:.3f}' for a in acc_a_all_after]}")

# Strategy B: Restore T0 using Coalition A+B as teacher
print(f"\n  ── Strategy B: Coalition A+B → restore cross-domain density ──")
tf_b = Transformer()
for tid in range(N_TASKS):
    tf_b.train(X_train[tid], y_train[tid])
acc_b_before = tf_b.accuracy(X_test[0], y_test[0])

# Coalition teacher: average of expert A and B probabilities
coalition_probs = (experts[0].prob_class1(X_train[0]) + experts[1].prob_class1(X_train[0])) / 2
tf_b.train_with_teacher(X_train[0], coalition_probs, epochs=400)

acc_b_after = tf_b.accuracy(X_test[0], y_test[0])
acc_b_all_after = [tf_b.accuracy(X_test[t], y_test[t]) for t in range(N_TASKS)]
print(f"    T0 accuracy: {acc_b_before:.3f} → {acc_b_after:.3f}  Δ={acc_b_after - acc_b_before:+.3f}")
print(f"    All tasks after restore: {[f'{a:.3f}' for a in acc_b_all_after]}")

# Strategy C: Restore using Compiled AB as teacher
print(f"\n  ── Strategy C: Compiled AB → restore joint density ──")
tf_c = Transformer()
for tid in range(N_TASKS):
    tf_c.train(X_train[tid], y_train[tid])
acc_c_before = tf_c.accuracy(X_test[0], y_test[0])

compiled_probs = compiled_AB.prob_class1(X_train[0])
tf_c.train_with_teacher(X_train[0], compiled_probs, epochs=400)

acc_c_after = tf_c.accuracy(X_test[0], y_test[0])
acc_c_all_after = [tf_c.accuracy(X_test[t], y_test[t]) for t in range(N_TASKS)]
print(f"    T0 accuracy: {acc_c_before:.3f} → {acc_c_after:.3f}  Δ={acc_c_after - acc_c_before:+.3f}")
print(f"    Compiled AB ref accuracy on T0: {compiled_AB.accuracy(X_test[0], y_test[0]):.3f}")
print(f"    All tasks after restore: {[f'{a:.3f}' for a in acc_c_all_after]}")

# Strategy D: Restore ALL forgotten tasks using respective experts
print(f"\n  ── Strategy D: Full restoration — all experts → all tasks ──")
tf_d = Transformer()
for tid in range(N_TASKS):
    tf_d.train(X_train[tid], y_train[tid])

acc_d_before = [tf_d.accuracy(X_test[t], y_test[t]) for t in range(N_TASKS)]
density_d_before = [tf_d.log_density(X_test[t]) for t in range(N_TASKS)]

# Restore each task using its expert
for tid in range(N_TASKS):
    teacher_probs = experts[tid].prob_class1(X_train[tid])
    tf_d.train_with_teacher(X_train[tid], teacher_probs, epochs=300)

acc_d_after = [tf_d.accuracy(X_test[t], y_test[t]) for t in range(N_TASKS)]
density_d_after = [tf_d.log_density(X_test[t]) for t in range(N_TASKS)]

print(f"    {'Task':<8} {'Before':>8} {'After':>8} {'Expert ref':>12} {'Recovered':>10} {'Density Δ':>10}")
print(f"    {'─'*58}")
for tid in range(N_TASKS):
    recovered = acc_d_after[tid] - acc_d_before[tid]
    d_delta = density_d_after[tid] - density_d_before[tid]
    print(f"    T{tid:<7} {acc_d_before[tid]:>8.3f} {acc_d_after[tid]:>8.3f} {experts[tid].accuracy(X_test[tid], y_test[tid]):>12.3f} {recovered:>+10.3f} {d_delta:>+10.4f}")
print(f"    Avg:  {np.mean(acc_d_before):.3f} → {np.mean(acc_d_after):.3f}  Δ={np.mean(acc_d_after) - np.mean(acc_d_before):+.3f}")

# ─── SUMMARY ───
print(f"\n{'='*70}")
print(f"SUMMARY: Density Restoration Strategies")
print(f"{'='*70}")
print(f"  {'Strategy':<30} {'T0 Before':>10} {'T0 After':>10} {'Δ':>10} {'Expert ref':>12}")
print(f"  {'─'*72}")
print(f"  {'No restoration':<30} {tf_final_accs[0]:>10.3f} {'—':>10} {'—':>10} {experts[0].accuracy(X_test[0], y_test[0]):>12.3f}")
print(f"  {'Solo Expert A':<30} {acc_a_before:>10.3f} {acc_a_after:>10.3f} {acc_a_after - acc_a_before:>+10.3f} {experts[0].accuracy(X_test[0], y_test[0]):>12.3f}")
print(f"  {'Coalition A+B':<30} {acc_b_before:>10.3f} {acc_b_after:>10.3f} {acc_b_after - acc_b_before:>+10.3f} {'—':>12}")
print(f"  {'Compiled AB':<30} {acc_c_before:>10.3f} {acc_c_after:>10.3f} {acc_c_after - acc_c_before:>+10.3f} {compiled_AB.accuracy(X_test[0], y_test[0]):>12.3f}")
print(f"  {'Full restoration (all)':<30} {np.mean(acc_d_before):>10.3f} {np.mean(acc_d_after):>10.3f} {np.mean(acc_d_after) - np.mean(acc_d_before):>+10.3f} {'—':>12}")
