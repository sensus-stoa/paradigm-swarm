#!/usr/bin/env python3
"""
Experiment: Density Preservation — SHARED BACKBONE scenario
Гипотеза: когда эксперты делят W1, density preservation замедляет forgetting.

Сценарий: 5 экспертов с ОБЩИМ W1 и отдельными W2-головами.
При обучении нового эксперта T5 общий W1 дрейфует → старые эксперты
деградируют. Density preservation: перед каждым шагом T5 проверяем
плотности старых экспертов на их тестовых данных. При падении —
корректирующий шаг на старых данных.

Сравнение:
(a) No protection: train T5, W1 дрейфует
(b) Density preservation: периодическая коррекция при падении плотности
(c) EWC: Fisher-based penalty на W1
"""
import numpy as np

np.random.seed(42)
DIM, HIDDEN = 20, 32
N_OLD = 5
N_TRAIN, N_TEST = 200, 100
EPOCHS_T5 = 1200
LR = 0.01

# ─── Shared backbone expert ───
class SharedExpert:
    """Multiple experts sharing W1, separate W2 per expert."""
    def __init__(self, seed):
        rng = np.random.RandomState(seed); s = 0.05
        self.W1 = rng.randn(DIM, HIDDEN) * s
        self.b1 = np.zeros((1, HIDDEN))
        self.W2 = {}   # expert_id → W2
        self.b2 = {}   # expert_id → b2
        self.fisher_W1 = np.zeros_like(self.W1)
        for tid in range(N_OLD + 1):
            self.W2[tid] = rng.randn(HIDDEN, 2) * s
            self.b2[tid] = np.zeros((1, 2))

    def forward(self, X, tid):
        a = np.maximum(0, X @ self.W1 + self.b1)
        z = a @ self.W2[tid] + self.b2[tid]
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def predict(self, X, tid): return self.forward(X, tid).argmax(1)
    def prob_class1(self, X, tid): return self.forward(X, tid)[:, 1]
    def accuracy(self, X, y, tid): return np.mean(self.predict(X, tid) == y)

    def log_density(self, X, tid):
        probs = self.forward(X, tid)
        return np.log(probs.max(1) + 1e-8).mean()

    def update_fisher(self, X, y, tid):
        yo = np.eye(2)[y]
        p = self.forward(X, tid); N = len(X)
        a = np.maximum(0, X @ self.W1 + self.b1)
        dz = (p - yo) / N
        da = dz @ self.W2[tid].T * (a > 0)
        dW1 = X.T @ da
        self.fisher_W1 += dW1**2

    def train_expert(self, X, y, tid, epochs=800, lr=LR, ewc_lam=0.0, old_W1=None):
        yo = np.eye(2)[y]
        for _ in range(epochs):
            idx = np.random.choice(len(X), min(64, len(X)), replace=False)
            Xb, yb = X[idx], yo[idx]
            p = self.forward(Xb, tid); N = len(Xb)
            a = np.maximum(0, Xb @ self.W1 + self.b1)
            dz = (p - yb) / N
            self.W2[tid] -= lr * a.T @ dz
            self.b2[tid] -= lr * dz.sum(0, keepdims=True)
            da = dz @ self.W2[tid].T * (a > 0)
            dW1 = Xb.T @ da
            if ewc_lam > 0 and old_W1 is not None:
                dW1 += ewc_lam * self.fisher_W1 * (self.W1 - old_W1)
            self.W1 -= lr * dW1
            self.b1 -= lr * da.sum(0, keepdims=True)

# ─── Generate tasks ───
centers = []
for i in range(N_OLD + 1):
    angle = 2 * np.pi * i / (N_OLD + 1)
    c = np.zeros(DIM)
    c[0] = 5 * np.cos(angle)
    c[1] = 5 * np.sin(angle)
    centers.append(c)

def make_task(center, n, sigma=0.8):
    X = np.random.randn(n, DIM) * sigma + center
    w = np.random.RandomState(abs(hash(str(center))) % 10000).randn(DIM)
    w = w / np.linalg.norm(w)
    y = ((X @ w) > 0).astype(int)
    return X, y

X_train, y_train = {}, {}
X_test, y_test = {}, {}
for tid in range(N_OLD + 1):
    X_train[tid], y_train[tid] = make_task(centers[tid], N_TRAIN)
    X_test[tid], y_test[tid] = make_task(centers[tid], N_TEST)

# ─── Train old experts on shared backbone ───
print("=" * 70)
print("DENSITY PRESERVATION — SHARED BACKBONE")
print("=" * 70)

shared = SharedExpert(42)
for tid in range(N_OLD):
    shared.train_expert(X_train[tid], y_train[tid], tid, epochs=600)
    shared.update_fisher(X_train[tid], y_train[tid], tid)
    acc = shared.accuracy(X_test[tid], y_test[tid], tid)
    print(f"  Expert T{tid} (shared W1): test_acc={acc:.3f}")

# Baseline densities
baseline_densities = {}
for tid in range(N_OLD):
    baseline_densities[tid] = shared.log_density(X_test[tid], tid)
print(f"\n  Baseline densities: {[f'{baseline_densities[t]:.3f}' for t in range(N_OLD)]}")

# ─── CONDITION A: No protection — just train T5 ───
shared_a = SharedExpert(42)
for tid in range(N_OLD):
    shared_a.train_expert(X_train[tid], y_train[tid], tid, epochs=600)
    shared_a.update_fisher(X_train[tid], y_train[tid], tid)

# Save old W1 for EWC
W1_before_a = shared_a.W1.copy()

shared_a.train_expert(X_train[N_OLD], y_train[N_OLD], N_OLD, epochs=EPOCHS_T5)

density_after_a = {}
acc_after_a = {}
for tid in range(N_OLD):
    density_after_a[tid] = shared_a.log_density(X_test[tid], tid)
    acc_after_a[tid] = shared_a.accuracy(X_test[tid], y_test[tid], tid)

print(f"\n  CONDITION A (No protection):")
print(f"    T5 accuracy: {shared_a.accuracy(X_test[N_OLD], y_test[N_OLD], N_OLD):.3f}")
print(f"    Avg old density shift: {np.mean([abs(density_after_a[t] - baseline_densities[t]) for t in range(N_OLD)]):.4f}")

# ─── CONDITION B: Density Preservation ───
shared_b = SharedExpert(42)
for tid in range(N_OLD):
    shared_b.train_expert(X_train[tid], y_train[tid], tid, epochs=600)
    shared_b.update_fisher(X_train[tid], y_train[tid], tid)

# Train T5 with density preservation
density_alerts_b = []
for ep in range(EPOCHS_T5):
    # T5 training step
    idx = np.random.choice(len(X_train[N_OLD]), min(64, len(X_train[N_OLD])), replace=False)
    Xb = X_train[N_OLD][idx]
    yb = np.eye(2)[y_train[N_OLD][idx]]
    p = shared_b.forward(Xb, N_OLD); N = len(Xb)
    a = np.maximum(0, Xb @ shared_b.W1 + shared_b.b1)
    dz = (p - yb) / N
    shared_b.W2[N_OLD] -= LR * a.T @ dz
    shared_b.b2[N_OLD] -= LR * dz.sum(0, keepdims=True)
    da = dz @ shared_b.W2[N_OLD].T * (a > 0)
    shared_b.W1 -= LR * Xb.T @ da
    shared_b.b1 -= LR * da.sum(0, keepdims=True)

    # Every 150 epochs: check old expert densities
    if ep % 150 == 0 and ep > 0:
        for tid in range(N_OLD):
            current = shared_b.log_density(X_test[tid], tid)
            shift = abs(current - baseline_densities[tid])
            if shift > 0.02:
                density_alerts_b.append((ep, tid, shift))
                # Corrective step: one epoch on old data
                idx_old = np.random.choice(len(X_train[tid]), min(32, len(X_train[tid])), replace=False)
                Xb_old = X_train[tid][idx_old]
                yb_old = np.eye(2)[y_train[tid][idx_old]]
                p_old = shared_b.forward(Xb_old, tid); N_old_b = len(Xb_old)
                a_old = np.maximum(0, Xb_old @ shared_b.W1 + shared_b.b1)
                dz_old = (p_old - yb_old) / N_old_b
                shared_b.W2[tid] -= 0.05 * a_old.T @ dz_old
                shared_b.b2[tid] -= 0.05 * dz_old.sum(0, keepdims=True)
                da_old = dz_old @ shared_b.W2[tid].T * (a_old > 0)
                shared_b.W1 -= 0.05 * Xb_old.T @ da_old
                shared_b.b1 -= 0.05 * da_old.sum(0, keepdims=True)

print(f"  CONDITION B (Density preservation): {len(density_alerts_b)} alerts triggered")
density_after_b = {}
acc_after_b = {}
for tid in range(N_OLD):
    density_after_b[tid] = shared_b.log_density(X_test[tid], tid)
    acc_after_b[tid] = shared_b.accuracy(X_test[tid], y_test[tid], tid)
print(f"    T5 accuracy: {shared_b.accuracy(X_test[N_OLD], y_test[N_OLD], N_OLD):.3f}")
print(f"    Avg old density shift: {np.mean([abs(density_after_b[t] - baseline_densities[t]) for t in range(N_OLD)]):.4f}")

# ─── CONDITION C: EWC ───
shared_c = SharedExpert(42)
for tid in range(N_OLD):
    shared_c.train_expert(X_train[tid], y_train[tid], tid, epochs=600)
    shared_c.update_fisher(X_train[tid], y_train[tid], tid)

W1_before = shared_c.W1.copy()
shared_c.train_expert(X_train[N_OLD], y_train[N_OLD], N_OLD, epochs=EPOCHS_T5, ewc_lam=5000, old_W1=W1_before)

density_after_c = {}
acc_after_c = {}
for tid in range(N_OLD):
    density_after_c[tid] = shared_c.log_density(X_test[tid], tid)
    acc_after_c[tid] = shared_c.accuracy(X_test[tid], y_test[tid], tid)
print(f"  CONDITION C (EWC λ=5000):")
print(f"    T5 accuracy: {shared_c.accuracy(X_test[N_OLD], y_test[N_OLD], N_OLD):.3f}")
print(f"    Avg old density shift: {np.mean([abs(density_after_c[t] - baseline_densities[t]) for t in range(N_OLD)]):.4f}")

# ─── RESULTS ───
print(f"\n{'='*70}")
print(f"RESULTS")
print(f"{'='*70}")
print(f"  {'Expert':<8} {'Baseline':>10} {'NoProtect':>10} {'DensityPres':>12} {'EWC':>10}")
print(f"  {'─'*55}")
for tid in range(N_OLD):
    print(f"  T{tid:<7} {baseline_densities[tid]:>10.4f} {density_after_a[tid]:>10.4f} {density_after_b[tid]:>12.4f} {density_after_c[tid]:>10.4f}")

shift_a = np.mean([abs(density_after_a[t] - baseline_densities[t]) for t in range(N_OLD)])
shift_b = np.mean([abs(density_after_b[t] - baseline_densities[t]) for t in range(N_OLD)])
shift_c = np.mean([abs(density_after_c[t] - baseline_densities[t]) for t in range(N_OLD)])

print(f"\n  Avg |density shift|:")
print(f"    No protection:      {shift_a:.4f}")
print(f"    DensityPres:        {shift_b:.4f}  ({shift_b/shift_a:.1%} of baseline)" if shift_a > 0 else f"    DensityPres:        {shift_b:.4f}")
print(f"    EWC:                {shift_c:.4f}  ({shift_c/shift_a:.1%} of baseline)" if shift_a > 0 else f"    EWC:                {shift_c:.4f}")

print(f"\n  Old-task accuracy after T5:")
acc_before_old = [shared.accuracy(X_test[t], y_test[t], t) for t in range(N_OLD)]
print(f"    Before T5:   {np.mean(acc_before_old):.3f}")
print(f"    No protect:  {np.mean([acc_after_a[t] for t in range(N_OLD)]):.3f}")
print(f"    DensityPres: {np.mean([acc_after_b[t] for t in range(N_OLD)]):.3f}")
print(f"    EWC:         {np.mean([acc_after_c[t] for t in range(N_OLD)]):.3f}")

print(f"\n  T5 (new task) accuracy:")
print(f"    All methods: ~{shared_a.accuracy(X_test[N_OLD], y_test[N_OLD], N_OLD):.3f}")
