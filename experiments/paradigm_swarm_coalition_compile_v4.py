#!/usr/bin/env python3
"""
Experiment 1 v4: Coalition Compilation — TRUE LABELS
Гипотеза: AB-плотность содержит структуру, невидимую для A и B по отдельности.
Скомпилированный эксперт, обученный на ИСТИННЫХ метках AB-данных,
превосходит и solo-экспертов, и коалицию.

Дизайн:
- A обучен на фичах [0-11], B на [8-19], 4 overlap
- AB-данные: активированы ОБА набора фичей, нелинейная целевая
- Сравнение: A, B, C(контроль), coalition(A+B), compiled(истинные метки), oracle

Отличие от v3: compiled получает НАСТОЯЩИЕ y_test, не pseudo-labels.
"""
import numpy as np

np.random.seed(42)
DIM, HIDDEN = 60, 32
N_TRAIN, N_AB, N_TEST = 300, 500, 400

FEATS_A = list(range(0, 12))
FEATS_B = list(range(8, 20))
FEATS_C = list(range(30, 42))

def make_solo(feats, n):
    X = np.random.randn(n, DIM) * 0.3
    X[:, feats] *= 6
    y = ((X[:, feats].sum(1)) > 0).astype(int)
    return X + np.random.randn(*X.shape) * 0.5, y

# ─── AB-распределения (НЕЛИНЕЙНЫЕ) ───
def make_ab_xor(n):
    X = np.random.randn(n, DIM) * 0.3
    X[:, FEATS_A] *= 6; X[:, FEATS_B] *= 6
    sa = (X[:, FEATS_A].sum(1) > 0).astype(int)
    sb = (X[:, FEATS_B].sum(1) > 0).astype(int)
    y = (sa ^ sb).astype(int)
    return X + np.random.randn(*X.shape) * 0.5, y

def make_ab_product(n):
    X = np.random.randn(n, DIM) * 0.3
    X[:, FEATS_A] *= 6; X[:, FEATS_B] *= 6
    sa = X[:, FEATS_A].sum(1); sb = X[:, FEATS_B].sum(1)
    y = ((sa * sb) > 0).astype(int)
    return X + np.random.randn(*X.shape) * 0.5, y

def make_ab_and(n):
    """y=1 только если ОБА сигнала положительны"""
    X = np.random.randn(n, DIM) * 0.3
    X[:, FEATS_A] *= 6; X[:, FEATS_B] *= 6
    y = ((X[:, FEATS_A].sum(1) > 0) & (X[:, FEATS_B].sum(1) > 0)).astype(int)
    return X + np.random.randn(*X.shape) * 0.5, y

def make_ab_nonlinear_sum(n):
    """y = sign(tanh(sumA) + tanh(sumB)) — нелинейная комбинация"""
    X = np.random.randn(n, DIM) * 0.3
    X[:, FEATS_A] *= 6; X[:, FEATS_B] *= 6
    sa = np.tanh(X[:, FEATS_A].sum(1) * 0.3)
    sb = np.tanh(X[:, FEATS_B].sum(1) * 0.3)
    y = ((sa + sb) > 0).astype(int)
    return X + np.random.randn(*X.shape) * 0.5, y

# ─── Expert ───
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
    def train(self, X, y, epochs=1500, lr=0.01):
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
    def accuracy(self, X, y): return np.mean(self.predict(X) == y)

# ─── Train solo experts ───
print("=" * 70)
print("EXPERIMENT 1 v4: TRUE LABELS — can AB expert see what A,B can't?")
print("=" * 70)

EA = Expert(42); EB = Expert(43); EC = Expert(44)
X_a, y_a = make_solo(FEATS_A, N_TRAIN); EA.train(X_a, y_a)
X_b, y_b = make_solo(FEATS_B, N_TRAIN); EB.train(X_b, y_b)
X_c, y_c = make_solo(FEATS_C, N_TRAIN); EC.train(X_c, y_c)

print(f"  Expert A (feats 0-11):    solo={EA.accuracy(X_a, y_a):.3f}")
print(f"  Expert B (feats 8-19):    solo={EB.accuracy(X_b, y_b):.3f}")
print(f"  Expert C (feats 30-41):   solo={EC.accuracy(X_c, y_c):.3f}")

# ─── TEST on AB distributions ───
distributions = [
    ("XOR",           make_ab_xor),
    ("PRODUCT",       make_ab_product),
    ("AND",           make_ab_and),
    ("TANH SUM",      make_ab_nonlinear_sum),
]

results = []

for dist_name, dist_fn in distributions:
    print(f"\n{'─'*70}")
    print(f"  {dist_name}")
    print(f"{'─'*70}")

    X_ab_train, y_ab_train = dist_fn(N_AB)
    X_ab_test, y_ab_test = dist_fn(N_TEST)

    # Solo accuracy
    acc_a = EA.accuracy(X_ab_test, y_ab_test)
    acc_b = EB.accuracy(X_ab_test, y_ab_test)
    acc_c = EC.accuracy(X_ab_test, y_ab_test)

    # Coalition (average probabilities)
    coal_avg = np.mean(((EA.prob_class1(X_ab_test) + EB.prob_class1(X_ab_test)) / 2 > 0.5).astype(int) == y_ab_test)
    coal_max = np.mean((np.maximum(EA.prob_class1(X_ab_test), EB.prob_class1(X_ab_test)) > 0.5).astype(int) == y_ab_test)

    # ─── Compiled AB: TRUE LABELS on AB training data ───
    comp_ab = Expert(99)
    comp_ab.train(X_ab_train, y_ab_train, epochs=2000)
    comp_acc = comp_ab.accuracy(X_ab_test, y_ab_test)

    # Oracle: trained + tested on AB (upper bound)
    oracle = Expert(77)
    oracle.train(X_ab_test[:N_TEST//2], y_ab_test[:N_TEST//2], epochs=2000)
    oracle_acc = oracle.accuracy(X_ab_test[N_TEST//2:], y_ab_test[N_TEST//2:])

    baseline = max(y_ab_test.mean(), 1 - y_ab_test.mean())

    print(f"  Baseline (chance):     {baseline:.3f}")
    print(f"  {'Predictor':<25} {'Acc':>8}  {'vs Chance':>10}  {'vs Best Solo':>12}  {'vs Coalition':>14}")
    print(f"  {'─'*70}")
    print(f"  {'Expert A (solo)':<25} {acc_a:>8.3f}  {'—':>10}  {'—':>12}  {'—':>14}")
    print(f"  {'Expert B (solo)':<25} {acc_b:>8.3f}")
    print(f"  {'Expert C (control)':<25} {acc_c:>8.3f}")
    best_solo = max(acc_a, acc_b, acc_c)
    print(f"  {'Coalition A+B (avg)':<25} {coal_avg:>8.3f}  {coal_avg - baseline:>+10.3f}  {coal_avg - best_solo:>+12.3f}  {'—':>14}")
    print(f"  {'Coalition A+B (max)':<25} {coal_max:>8.3f}")
    best_coalition = max(coal_avg, coal_max)
    print(f"  {'Compiled AB (TRUE)':<25} {comp_acc:>8.3f}  {comp_acc - baseline:>+10.3f}  {comp_acc - best_solo:>+12.3f}  {comp_acc - best_coalition:>+14.3f}")
    print(f"  {'Oracle':<25} {oracle_acc:>8.3f}")

    results.append({
        'name': dist_name,
        'best_solo': best_solo,
        'best_coalition': best_coalition,
        'compiled': comp_acc,
        'oracle': oracle_acc,
        'delta_solo': comp_acc - best_solo,
        'delta_coalition': comp_acc - best_coalition,
    })

# ─── ABLATION: training samples → compiled accuracy ───
print(f"\n{'='*70}")
print(f"ABLATION: AB training samples → Compiled accuracy (XOR)")
print(f"{'='*70}")
print(f"  {'Samples':>10} {'Compiled':>10} {'Coalition':>12} {'Best Solo':>12} {'Oracle':>10}")
print(f"  {'─'*55}")
X_abl_test, y_abl_test = make_ab_xor(N_TEST)
for n in [30, 60, 100, 200, 350, 500]:
    X_abl, y_abl = make_ab_xor(n)
    cabl = Expert(99 + n)
    cabl.train(X_abl, y_abl, epochs=1500)
    cabl_acc = cabl.accuracy(X_abl_test, y_abl_test)
    # Recompute coalition on this test
    ca = np.mean(((EA.prob_class1(X_abl_test)+EB.prob_class1(X_abl_test))/2 > 0.5).astype(int) == y_abl_test)
    bs = max(EA.accuracy(X_abl_test, y_abl_test), EB.accuracy(X_abl_test, y_abl_test))
    o = Expert(77)
    o.train(X_abl_test[:200], y_abl_test[:200], epochs=2000)
    oa = o.accuracy(X_abl_test[200:], y_abl_test[200:])
    print(f"  {n:>10} {cabl_acc:>10.3f} {ca:>12.3f} {bs:>12.3f} {oa:>10.3f}")

# ─── FINAL ───
print(f"\n{'='*70}")
print(f"FINAL: Compiled AB (TRUE LABELS) vs Solo vs Coalition")
print(f"{'='*70}")
print(f"  {'Distribution':<15} {'Best Solo':>10} {'Coalition':>10} {'Compiled':>10} {'Oracle':>10} {'Δ(Comp-Coal)':>14}")
print(f"  {'─'*70}")
wins = 0
for r in results:
    m = " ✓" if r['delta_coalition'] > 0.01 else ""
    wins += 1 if r['delta_coalition'] > 0.01 else 0
    print(f"  {r['name']:<15} {r['best_solo']:>10.3f} {r['best_coalition']:>10.3f} {r['compiled']:>10.3f} {r['oracle']:>10.3f} {r['delta_coalition']:>+14.3f}{m}")
print(f"\n  Compiled > Coalition in {wins}/{len(results)} distributions")
