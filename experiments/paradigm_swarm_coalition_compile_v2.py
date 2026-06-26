#!/usr/bin/env python3
"""
Experiment 1 v2: Coalition Compilation — NON-LINEAR interaction
Гипотеза: когда A и B взаимодействуют нелинейно (XOR/product), 
compiled эксперт бьёт коалицию.

v1 показал: при аддитивных признаках coalition avg > compiled.
Здесь: cross-domain target = XOR(A_features, B_features) — 
каждый эксперт по отдельности видит шум (~50%), коалиция тоже,
только compiled ловит joint pattern.
"""
import numpy as np

np.random.seed(42)
DIM, HIDDEN = 60, 32
N_TRAIN = 300
N_CROSS = 400

FEATURES = {
    0: list(range(0, 12)),                    # A
    1: list(range(8, 20)),                     # B (4 overlap with A)
    2: list(range(30, 42)),                    # C (isolated control)
}

# ─── Generate training data for solo experts ───
def make_solo(feats, n):
    X = np.random.randn(n, DIM) * 0.3
    X[:, feats] *= 6
    y = ((X[:, feats].sum(1)) > 0).astype(int)
    X += np.random.randn(*X.shape) * 0.5
    return X, y

# ─── Generate CROSS-DOMAIN data: NON-LINEAR interaction ───
def make_cross_xor(feats_a, feats_b, n):
    """
    y = 1 if (sum(A) > 0) XOR (sum(B) > 0), else 0.
    Each expert alone sees noise (~50% accuracy).
    Coalition averaging sees ~50%.
    Only a model trained on BOTH can learn the XOR pattern.
    """
    X = np.random.randn(n, DIM) * 0.3
    X[:, feats_a] *= 6
    X[:, feats_b] *= 6
    sig_a = (X[:, feats_a].sum(1) > 0).astype(int)
    sig_b = (X[:, feats_b].sum(1) > 0).astype(int)
    y = (sig_a ^ sig_b).astype(int)  # XOR
    X += np.random.randn(*X.shape) * 0.5
    return X, y

def make_cross_product(feats_a, feats_b, n):
    """
    y = 1 if sum(A) * sum(B) > 0, else 0.
    Both must agree direction — AND-like nonlinearity.
    """
    X = np.random.randn(n, DIM) * 0.3
    X[:, feats_a] *= 6
    X[:, feats_b] *= 6
    sig_a = X[:, feats_a].sum(1)
    sig_b = X[:, feats_b].sum(1)
    y = ((sig_a * sig_b) > 0).astype(int)
    X += np.random.randn(*X.shape) * 0.5
    return X, y

def make_cross_threshold(feats_a, feats_b, n):
    """
    y = 1 only when BOTH sum(A) > t AND sum(B) > t.
    Each expert alone is useless (can't see the other condition).
    """
    X = np.random.randn(n, DIM) * 0.3
    X[:, feats_a] *= 6
    X[:, feats_b] *= 6
    t_a = np.percentile(X[:, feats_a].sum(1), 50)
    t_b = np.percentile(X[:, feats_b].sum(1), 50)
    y = ((X[:, feats_a].sum(1) > t_a) & (X[:, feats_b].sum(1) > t_b)).astype(int)
    X += np.random.randn(*X.shape) * 0.5
    return X, y

# ─── Expert class ───
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
    def train(self, X, y, epochs=1200, lr=0.01):
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

# ─── Train solo experts ───
print("=" * 70)
print("EXPERIMENT 1 v2: Coalition Compilation — NON-LINEAR")
print("=" * 70)

X_solo, y_solo = {}, {}
experts = {}
for tid in range(3):
    X_solo[tid], y_solo[tid] = make_solo(FEATURES[tid], N_TRAIN)
    e = Expert(42 + tid)
    e.train(X_solo[tid], y_solo[tid])
    experts[tid] = e
    acc = e.accuracy(X_solo[tid], y_solo[tid])
    print(f"  Expert {tid}: solo_acc={acc:.3f}  feats={FEATURES[tid]}")

# ─── Compiled AB: train on combined A+B solo data (not cross!) ───
X_AB_union = np.vstack([X_solo[0], X_solo[1]])
y_AB_union = np.hstack([y_solo[0], y_solo[1]])
compiled_AB = Expert(99)
compiled_AB.train(X_AB_union, y_AB_union, epochs=2000)
print(f"  Compiled AB: union_acc={compiled_AB.accuracy(X_AB_union, y_AB_union):.3f}")

# ─── Test on cross-domain distributions ───
X_test_xor, y_test_xor = make_cross_xor(FEATURES[0], FEATURES[1], N_CROSS)
X_test_prod, y_test_prod = make_cross_product(FEATURES[0], FEATURES[1], N_CROSS)
X_test_thr, y_test_thr = make_cross_threshold(FEATURES[0], FEATURES[1], N_CROSS)

def coalition_predict(X, eids, strategy='avg'):
    if strategy == 'vote':
        preds = np.column_stack([experts[eid].predict(X) for eid in eids])
        combined = (preds.mean(1) > 0.5).astype(float)
    else:
        probs = np.column_stack([experts[eid].prob_class1(X) for eid in eids])
        if strategy == 'avg':
            combined = probs.mean(1)
        else:
            combined = probs.max(1)
    return (combined > 0.5).astype(int)

def run_test(name, X_test, y_test):
    print(f"\n{'─'*70}")
    print(f"  {name}")
    print(f"{'─'*70}")
    print(f"  Baseline (chance): {max(y_test.mean(), 1-y_test.mean()):.3f}")

    # Solo experts
    solo = {}
    for tid in range(3):
        solo[tid] = experts[tid].accuracy(X_test, y_test)
    best_solo = max(solo.items(), key=lambda x: x[1])

    # Coalition
    coal_avg = np.mean(coalition_predict(X_test, [0, 1], 'avg') == y_test)
    coal_max = np.mean(coalition_predict(X_test, [0, 1], 'max') == y_test)

    # Compiled
    comp_acc = compiled_AB.accuracy(X_test, y_test)

    # Oracle
    oracle = Expert(77)
    oracle.train(X_test[:N_CROSS//2], y_test[:N_CROSS//2], epochs=2000)
    oracle_acc = oracle.accuracy(X_test[N_CROSS//2:], y_test[N_CROSS//2:])

    print(f"  {'Method':<30} {'Acc':>8}  {'vs Chance':>10}  {'vs Best Solo':>12}")
    print(f"  {'─'*60}")
    print(f"  {'Best solo (E{})'.format(best_solo[0]):<30} {best_solo[1]:>8.3f}  {'—':>10}  {'—':>12}")
    print(f"  {'Coalition A+B (avg)':<30} {coal_avg:>8.3f}  {coal_avg - max(y_test.mean(), 1-y_test.mean()):>+10.3f}  {coal_avg - best_solo[1]:>+12.3f}")
    print(f"  {'Coalition A+B (max)':<30} {coal_max:>8.3f}")
    print(f"  {'Compiled AB':<30} {comp_acc:>8.3f}  {comp_acc - max(y_test.mean(), 1-y_test.mean()):>+10.3f}  {comp_acc - best_solo[1]:>+12.3f}")
    print(f"  {'Oracle (upper bound)':<30} {oracle_acc:>8.3f}")

    best_coalition = max(coal_avg, coal_max)
    return {
        'name': name,
        'compiled': comp_acc,
        'best_coalition': best_coalition,
        'best_solo': best_solo[1],
        'oracle': oracle_acc,
        'delta_vs_coalition': comp_acc - best_coalition,
    }

results = []
results.append(run_test("XOR: y = A_features XOR B_features", X_test_xor, y_test_xor))
results.append(run_test("PRODUCT: y = sign(A·B)", X_test_prod, y_test_prod))
results.append(run_test("THRESHOLD: y = (A>t AND B>t)", X_test_thr, y_test_thr))

# ─── SUMMARY ───
print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"  {'Distribution':<20} {'Compiled':>10} {'Best Coal':>10} {'Best Solo':>10} {'Δ(Comp-Coal)':>14}")
print(f"  {'─'*65}")
for r in results:
    print(f"  {r['name']:<20} {r['compiled']:>10.3f} {r['best_coalition']:>10.3f} {r['best_solo']:>10.3f} {r['delta_vs_coalition']:>+14.3f}")
print(f"\n  Hypothesis confirmed: compiled > coalition for non-linear interactions" if 
      all(r['delta_vs_coalition'] > 0 for r in results) else
      f"  Mixed: {sum(1 for r in results if r['delta_vs_coalition'] > 0)}/{len(results)} in favour of compiled")
