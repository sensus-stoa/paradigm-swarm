#!/usr/bin/env python3
"""
Experiment 1: Coalition Compilation
Гипотеза: скомпилированный эксперт AB точнее коалиции A+B на кросс-доменных случаях.

Дизайн:
- 5 экспертов с пересекающимися feature spaces
- Кросс-доменный тест: признаки A И B активны одновременно
- Сравнение: Single / Coalition(A+B) / Compiled(AB) / Oracle
"""
import numpy as np

np.random.seed(42)
DIM, HIDDEN = 60, 32
N_TRAIN = [200, 200, 200, 200, 200]
N_CROSS = 300  # cross-domain test samples

# ─── FEATURE SPACES (overlapping subsets) ───
FEATURES = {
    0: list(range(0, 12)),                    # A: shape
    1: list(range(6, 18)),                     # B: shape+texture (6 overlap with A)
    2: list(range(12, 24)),                    # C: texture+color (6 overlap with B)
    3: list(range(0, 6)) + list(range(12, 18)),  # D: A∩C (no direct overlap, different combo)
    4: list(range(30, 42)),                    # E: isolated (pattern)
}

# ─── Generate data ───
def make_task(feats, n, sign=1):
    X = np.random.randn(n, DIM) * 0.3
    X[:, feats] *= 6
    y = ((X[:, feats].sum(1) * sign) > 0).astype(int)
    X += np.random.randn(*X.shape) * 0.5
    return X, y

X_train, y_train = {}, {}
for tid in range(5):
    X_train[tid], y_train[tid] = make_task(FEATURES[tid], N_TRAIN[tid])

# Cross-domain test: A AND B features active
X_cross_AB, y_cross_AB = make_task(FEATURES[0] + FEATURES[1], N_CROSS)

# Cross-domain test: B AND C features active
X_cross_BC, y_cross_BC = make_task(FEATURES[1] + FEATURES[2], N_CROSS)

# Cross-domain test: D features (A∩C pattern — already a "compiled" concept)
X_cross_D, y_cross_D = make_task(FEATURES[3], N_CROSS)

# ─── Expert (same architecture as benchmark) ───
class Expert:
    def __init__(self, seed):
        rng = np.random.RandomState(seed)
        s = 0.05
        self.W1 = rng.randn(DIM, HIDDEN) * s
        self.b1 = np.zeros((1, HIDDEN))
        self.W2 = rng.randn(HIDDEN, 2) * s
        self.b2 = np.zeros((1, 2))
    def forward(self, X):
        a = np.maximum(0, X @ self.W1 + self.b1)
        z = a @ self.W2 + self.b2
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
    def predict(self, X):
        return self.forward(X).argmax(1)
    def prob_class1(self, X):
        return self.forward(X)[:, 1]
    def train(self, X, y, epochs=800, lr=0.01):
        yo = np.eye(2)[y]
        for _ in range(epochs):
            idx = np.random.choice(len(X), min(64, len(X)), replace=False)
            Xb, yb = X[idx], yo[idx]
            p = self.forward(Xb)
            N = len(Xb)
            a = np.maximum(0, Xb @ self.W1 + self.b1)
            dz = (p - yb) / N
            self.W2 -= lr * a.T @ dz
            self.b2 -= lr * dz.sum(0, keepdims=True)
            da = dz @ self.W2.T * (a > 0)
            self.W1 -= lr * Xb.T @ da
            self.b1 -= lr * da.sum(0, keepdims=True)
    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

# ─── Train independent experts ───
print("=" * 70)
print("COALITION COMPILATION EXPERIMENT")
print("=" * 70)

experts = {}
for tid in range(5):
    e = Expert(42 + tid)
    e.train(X_train[tid], y_train[tid], epochs=1000)
    experts[tid] = e
    acc = e.accuracy(X_train[tid], y_train[tid])
    print(f"  Expert {tid}: train_acc={acc:.3f}  features={FEATURES[tid]}")

# ─── Compiled expert AB: trained on A∪B data ───
# Uses BOTH feature sets — learns interaction
X_AB_union = np.vstack([X_train[0], X_train[1]])
y_AB_union = np.hstack([y_train[0], y_train[1]])
compiled_AB = Expert(99)
compiled_AB.train(X_AB_union, y_AB_union, epochs=1500)
acc_ab = compiled_AB.accuracy(X_AB_union, y_AB_union)
print(f"\n  Compiled AB: train_acc={acc_ab:.3f}  (A∪B, {len(X_AB_union)} samples)")

# Compiled BC
X_BC_union = np.vstack([X_train[1], X_train[2]])
y_BC_union = np.hstack([y_train[1], y_train[2]])
compiled_BC = Expert(98)
compiled_BC.train(X_BC_union, y_BC_union, epochs=1500)
acc_bc = compiled_BC.accuracy(X_BC_union, y_BC_union)
print(f"  Compiled BC: train_acc={acc_bc:.3f}  (B∪C, {len(X_BC_union)} samples)")

# ─── TEST: Cross-domain AB cases ───
print(f"\n{'='*70}")
print(f"TEST: Cross-domain AB (features from BOTH A and B active)")
print(f"{'='*70}")

# Coalition strategies
def coalition_predict(X, eids, strategy='avg'):
    """Combine experts by averaging or maxing class-1 probabilities"""
    if strategy == 'vote':
        preds = np.column_stack([experts[eid].predict(X) for eid in eids])
        combined = (preds.mean(1) > 0.5).astype(float)
    else:
        probs = np.column_stack([experts[eid].prob_class1(X) for eid in eids])
        if strategy == 'avg':
            combined = probs.mean(1)
        else:  # max
            combined = probs.max(1)
    return (combined > 0.5).astype(int)

# Single best expert on AB cross data
single_acc = {}
for tid in range(5):
    single_acc[tid] = experts[tid].accuracy(X_cross_AB, y_cross_AB)
best_single = max(single_acc.items(), key=lambda x: x[1])

# Coalition A+B
coalition_avg_acc = np.mean(coalition_predict(X_cross_AB, [0, 1], 'avg') == y_cross_AB)
coalition_max_acc = np.mean(coalition_predict(X_cross_AB, [0, 1], 'max') == y_cross_AB)
coalition_vote_acc = np.mean(coalition_predict(X_cross_AB, [0, 1], 'vote') == y_cross_AB)

# Compiled AB
compiled_ab_acc = compiled_AB.accuracy(X_cross_AB, y_cross_AB)

# Oracle: train directly on cross-domain data (upper bound)
oracle_AB = Expert(77)
oracle_AB.train(X_cross_AB[:200], y_cross_AB[:200], epochs=1500)
oracle_ab_acc = oracle_AB.accuracy(X_cross_AB[200:], y_cross_AB[200:])

print(f"\n  {'Method':<30} {'Accuracy':>10}")
print(f"  {'-'*40}")
print(f"  {'Best single expert (E{})'.format(best_single[0]):<30} {best_single[1]:>10.3f}")
print(f"  {'Coalition A+B (avg prob)':<30} {coalition_avg_acc:>10.3f}")
print(f"  {'Coalition A+B (max prob)':<30} {coalition_max_acc:>10.3f}")
print(f"  {'Coalition A+B (vote)':<30} {coalition_vote_acc:>10.3f}")
print(f"  {'Compiled AB':<30} {compiled_ab_acc:>10.3f}")
print(f"  {'Oracle (upper bound)':<30} {oracle_ab_acc:>10.3f}")
print(f"\n  Δ(Compiled - Best Coalition) = {compiled_ab_acc - max(coalition_avg_acc, coalition_max_acc, coalition_vote_acc):+.3f}")
print(f"  Δ(Compiled - Best Single)    = {compiled_ab_acc - best_single[1]:+.3f}")
print(f"  Δ(Oracle - Compiled)         = {oracle_ab_acc - compiled_ab_acc:+.3f}")

# ─── TEST: Cross-domain BC cases ───
print(f"\n{'='*70}")
print(f"TEST: Cross-domain BC (features from BOTH B and C active)")
print(f"{'='*70}")

best_single_bc = max((experts[tid].accuracy(X_cross_BC, y_cross_BC), tid) for tid in range(5))
coalition_bc_avg = np.mean(coalition_predict(X_cross_BC, [1, 2], 'avg') == y_cross_BC)
coalition_bc_max = np.mean(coalition_predict(X_cross_BC, [1, 2], 'max') == y_cross_BC)
compiled_bc_acc = compiled_BC.accuracy(X_cross_BC, y_cross_BC)

oracle_BC = Expert(76)
oracle_BC.train(X_cross_BC[:200], y_cross_BC[:200], epochs=1500)
oracle_bc_acc = oracle_BC.accuracy(X_cross_BC[200:], y_cross_BC[200:])

print(f"\n  {'Method':<30} {'Accuracy':>10}")
print(f"  {'-'*40}")
print(f"  {'Best single (E{})'.format(best_single_bc[1]):<30} {best_single_bc[0]:>10.3f}")
print(f"  {'Coalition B+C (avg)':<30} {coalition_bc_avg:>10.3f}")
print(f"  {'Coalition B+C (max)':<30} {coalition_bc_max:>10.3f}")
print(f"  {'Compiled BC':<30} {compiled_bc_acc:>10.3f}")
print(f"  {'Oracle (upper bound)':<30} {oracle_bc_acc:>10.3f}")
print(f"\n  Δ(Compiled - Best Coalition) = {compiled_bc_acc - max(coalition_bc_avg, coalition_bc_max):+.3f}")

# ─── ABLATION: How many coalition samples needed for compiled expert? ───
print(f"\n{'='*70}")
print(f"ABLATION: Coalition samples → Compiled accuracy")
print(f"{'='*70}")
print(f"  {'Samples':>10} {'Compiled AB':>15} {'Coalition avg':>15} {'Δ':>10}")
print(f"  {'-'*55}")

for n_samples in [50, 100, 200, 300, 400]:
    X_abl = np.vstack([X_train[0][:n_samples//2], X_train[1][:n_samples//2]])
    y_abl = np.hstack([y_train[0][:n_samples//2], y_train[1][:n_samples//2]])
    e_abl = Expert(99 + n_samples)
    e_abl.train(X_abl, y_abl, epochs=1000)
    comp_acc = e_abl.accuracy(X_cross_AB, y_cross_AB)
    print(f"  {n_samples:>10} {comp_acc:>15.3f} {coalition_avg_acc:>15.3f} {comp_acc - coalition_avg_acc:>+10.3f}")

# ─── SUMMARY ───
print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"  Coalition AB: compiled {compiled_ab_acc:.3f} vs best coalition {max(coalition_avg_acc, coalition_max_acc, coalition_vote_acc):.3f}")
print(f"  Coalition BC: compiled {compiled_bc_acc:.3f} vs best coalition {max(coalition_bc_avg, coalition_bc_max):.3f}")
print(f"  Both coalitions show {'improvement' if compiled_ab_acc > coalition_avg_acc and compiled_bc_acc > coalition_bc_avg else 'mixed results'}")
