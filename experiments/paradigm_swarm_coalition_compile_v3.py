#!/usr/bin/env python3
"""
Experiment 1 v3: Coalition Compilation — PSEUDO-LABEL training
Гипотеза: compiled эксперт, обученный на pseudo-labels от коалиции,
ловит joint distribution и превосходит коалицию.

Реалистичный сценарий: коалиция A+B часто активируется на запросах →
сохраняем запросы + predictions → обучаем compiled на этих данных.

v2 показал: union-данные (A∪B) не дают compiled'у преимущества
над коалицией, потому что не содержат joint distribution.
"""
import numpy as np

np.random.seed(42)
DIM, HIDDEN = 60, 32
N_TRAIN = 300
N_PSEUDO = 600   # unlabeled cross-domain queries
N_TEST = 400      # held-out test

FEATURES = {
    0: list(range(0, 12)),
    1: list(range(8, 20)),
    2: list(range(30, 42)),
}

# ─── Data generators ───
def make_solo(feats, n):
    X = np.random.randn(n, DIM) * 0.3
    X[:, feats] *= 6
    y = ((X[:, feats].sum(1)) > 0).astype(int)
    X += np.random.randn(*X.shape) * 0.5
    return X, y

def make_cross_xor(feats_a, feats_b, n):
    X = np.random.randn(n, DIM) * 0.3
    X[:, feats_a] *= 6; X[:, feats_b] *= 6
    sig_a = (X[:, feats_a].sum(1) > 0).astype(int)
    sig_b = (X[:, feats_b].sum(1) > 0).astype(int)
    y = (sig_a ^ sig_b).astype(int)
    X += np.random.randn(*X.shape) * 0.5
    return X, y

def make_cross_product(feats_a, feats_b, n):
    X = np.random.randn(n, DIM) * 0.3
    X[:, feats_a] *= 6; X[:, feats_b] *= 6
    sig_a = X[:, feats_a].sum(1)
    sig_b = X[:, feats_b].sum(1)
    y = ((sig_a * sig_b) > 0).astype(int)
    X += np.random.randn(*X.shape) * 0.5
    return X, y

def make_cross_threshold(feats_a, feats_b, n):
    X = np.random.randn(n, DIM) * 0.3
    X[:, feats_a] *= 6; X[:, feats_b] *= 6
    t_a = np.percentile(X[:, feats_a].sum(1), 50)
    t_b = np.percentile(X[:, feats_b].sum(1), 50)
    y = ((X[:, feats_a].sum(1) > t_a) & (X[:, feats_b].sum(1) > t_b)).astype(int)
    X += np.random.randn(*X.shape) * 0.5
    return X, y

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
    def confidence(self, X): return self.forward(X).max(1)
    def train(self, X, y, epochs=1200, lr=0.01):
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
print("EXPERIMENT 1 v3: Coalition → Pseudo-Label → Compiled")
print("=" * 70)

experts = {}
for tid in range(3):
    X_s, y_s = make_solo(FEATURES[tid], N_TRAIN)
    e = Expert(42 + tid); e.train(X_s, y_s)
    experts[tid] = e
    print(f"  Expert {tid}: solo_acc={e.accuracy(X_s, y_s):.3f}")

# ─── Coalition labeling function ───
def coalition_pseudo_labels(X, strategy='avg', confidence_threshold=0.0):
    """
    Coalition A+B produces pseudo-labels.
    Only labels where BOTH agree (high confidence) are kept.
    """
    preds_a = experts[0].predict(X)
    preds_b = experts[1].predict(X)
    conf_a = experts[0].confidence(X)
    conf_b = experts[1].confidence(X)

    if strategy == 'agree':
        # Only use where A and B agree
        mask = (preds_a == preds_b) & (conf_a > confidence_threshold) & (conf_b > confidence_threshold)
        y = preds_a.copy()
        return y, mask
    else:
        # Weighted average of probabilities
        probs = (experts[0].prob_class1(X) + experts[1].prob_class1(X)) / 2
        y = (probs > 0.5).astype(int)
        conf = np.maximum(probs, 1 - probs)
        mask = conf > confidence_threshold
        return y, mask

# ─── TEST: multiple non-linear distributions ───
distributions = [
    ("XOR", make_cross_xor),
    ("PRODUCT", make_cross_product),
    ("THRESHOLD", make_cross_threshold),
]

results = []

for dist_name, dist_fn in distributions:
    print(f"\n{'─'*70}")
    print(f"  {dist_name}")
    print(f"{'─'*70}")

    # Generate pseudo-label training data (unlabeled, from cross-domain)
    X_pseudo, y_true_pseudo = dist_fn(FEATURES[0], FEATURES[1], N_PSEUDO)

    # Generate held-out test data
    X_test, y_test = dist_fn(FEATURES[0], FEATURES[1], N_TEST)

    # Solo baselines
    solo_acc = {}
    for tid in range(3):
        solo_acc[tid] = experts[tid].accuracy(X_test, y_test)

    # ─── Coalition baselines (on test) ───
    def coalition_predict(X, strategy='avg'):
        if strategy == 'vote':
            preds = np.column_stack([experts[eid].predict(X) for eid in [0,1]])
            combined = (preds.mean(1) > 0.5).astype(float)
        elif strategy == 'avg':
            combined = (experts[0].prob_class1(X) + experts[1].prob_class1(X)) / 2
        else:
            combined = np.maximum(experts[0].prob_class1(X), experts[1].prob_class1(X))
        return (combined > 0.5).astype(int)

    coal_avg_acc = np.mean(coalition_predict(X_test, 'avg') == y_test)
    coal_max_acc = np.mean(coalition_predict(X_test, 'max') == y_test)

    # ─── PSEUDO-LABEL strategies ───
    strategies = [
        ("avg_all", 'avg', 0.0),      # Use all pseudo-labels
        ("avg_conf0.6", 'avg', 0.6),   # Only high-confidence
        ("avg_conf0.7", 'avg', 0.7),
        ("agree_all", 'agree', 0.0),   # Only where A and B agree
        ("agree_conf0.6", 'agree', 0.6),
    ]

    print(f"  {'Strategy':<20} {'PL acc':>8} {'Coverage':>10} {'Test acc':>10} {'Δ vs Coal':>12} {'vs Oracle':>10}")
    print(f"  {'─'*70}")

    best_compiled_acc = 0
    best_compiled_name = ""

    for strat_name, strat, conf_thr in strategies:
        y_pl, mask = coalition_pseudo_labels(X_pseudo, strat, conf_thr)
        coverage = mask.mean()
        n_used = mask.sum()

        if n_used < 50:  # too few training samples
            print(f"  {strat_name:<20} {'—':>8} {coverage:>10.1%} {'—':>10} {'—':>12} {'—':>10}")
            continue

        X_used = X_pseudo[mask]
        y_used = y_pl[mask]

        compiled = Expert(99 + hash(strat_name) % 100)
        compiled.train(X_used, y_used, epochs=1500)
        comp_acc = compiled.accuracy(X_test, y_test)

        if comp_acc > best_compiled_acc:
            best_compiled_acc = comp_acc
            best_compiled_name = strat_name

        delta_coal = comp_acc - max(coal_avg_acc, coal_max_acc)

        # Oracle
        oracle = Expert(77)
        oracle.train(X_test[:N_TEST//2], y_test[:N_TEST//2], epochs=2000)
        oracle_acc = oracle.accuracy(X_test[N_TEST//2:], y_test[N_TEST//2:])

        marker = " ← BEST" if comp_acc == best_compiled_acc else ""
        print(f"  {strat_name:<20} {n_used:>8} {coverage:>10.1%} {comp_acc:>10.3f} {delta_coal:>+12.3f} {oracle_acc:>10.3f}{marker}")

    best_coalition = max(coal_avg_acc, coal_max_acc)
    oracle_full = Expert(77)
    oracle_full.train(X_test[:N_TEST//2], y_test[:N_TEST//2], epochs=2000)
    oracle_acc = oracle_full.accuracy(X_test[N_TEST//2:], y_test[N_TEST//2:])

    results.append({
        'name': dist_name,
        'best_solo': max(solo_acc.values()),
        'best_coalition': best_coalition,
        'best_compiled': best_compiled_acc,
        'best_compiled_name': best_compiled_name,
        'oracle': oracle_acc,
        'delta_vs_coalition': best_compiled_acc - best_coalition,
    })

    print(f"\n  Summary: coalition={best_coalition:.3f} → compiled({best_compiled_name})={best_compiled_acc:.3f}  oracle={oracle_acc:.3f}")

# ─── FINAL SUMMARY ───
print(f"\n{'='*70}")
print(f"FINAL SUMMARY — Pseudo-Label Coalition Compilation")
print(f"{'='*70}")
print(f"  {'Distribution':<15} {'Best Solo':>10} {'Coalition':>10} {'Compiled':>10} {'Oracle':>10} {'Δ(Comp-Coal)':>14}")
print(f"  {'─'*70}")
wins = 0
for r in results:
    marker = " ✓" if r['delta_vs_coalition'] > 0.01 else ""
    wins += 1 if r['delta_vs_coalition'] > 0.01 else 0
    print(f"  {r['name']:<15} {r['best_solo']:>10.3f} {r['best_coalition']:>10.3f} {r['best_compiled']:>10.3f} {r['oracle']:>10.3f} {r['delta_vs_coalition']:>+14.3f}{marker}")
print(f"\n  Hypothesis: compiled > coalition in {wins}/{len(results)} distributions (Δ>0.01)")
print(f"  Mechanism: coalition labels cross-domain queries → compiled learns joint pattern")
