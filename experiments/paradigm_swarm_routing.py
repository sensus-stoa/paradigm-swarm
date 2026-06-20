#!/usr/bin/env python3
"""
END-TO-END ROUTING EXPERIMENT — EXPERTS ROUTE THEMSELVES (§4.18)

Closes the routing gap with the router-free principle (§4.17):
  NO separate router module.
  Each expert estimates query relevance → argmax = routing.

Three self-routing mechanisms compared:
  - confidence:  argmax_i softmax_confidence_i(x)   [experts' own outputs]
  - density:     argmax_i gaussian_density_i(x)      [experts model input distribution]
  - prototype:   argmin_i ||x - mean_train_i||       [baseline, nearest cluster center]

Degradation fix: early stopping per expert (track val accuracy, stop when it drops).

Setup: 4 separated 2D Gaussian clusters, MLP experts, mixed test queries.
"""
import numpy as np

np.random.seed(42)

# ═══════════════════════════════════════════════════
# SETUP: 4 well-separated 2D clusters + binary classification
# ═══════════════════════════════════════════════════
N_TASKS = 4
N_TRAIN, N_VAL, N_TEST = 200, 50, 100
HIDDEN, LR = 32, 0.05

CENTERS = {0: np.array([-4, -4]), 1: np.array([4, -4]),
           2: np.array([-4, 4]),  3: np.array([4, 4])}

# Decision hyperplanes through centers (binary classification)
task_w = {0: np.array([1.0, 0.5]), 1: np.array([0.5, -1.0]),
          2: np.array([-1.0, 0.5]), 3: np.array([-0.5, -1.0])}
for k in task_w:
    task_w[k] = task_w[k] / np.linalg.norm(task_w[k])

X_train, y_train = [], []
X_val,   y_val   = [], []
X_test,  y_test  = [], []

for tid in range(N_TASKS):
    c, w = CENTERS[tid], task_w[tid]

    X_tr = np.random.randn(N_TRAIN, 2) * 1.2 + c
    X_va = np.random.randn(N_VAL, 2)   * 1.2 + c
    X_te = np.random.randn(N_TEST, 2)  * 1.2 + c

    y_tr = ((X_tr - c) @ w > 0).astype(int)
    y_va = ((X_va - c) @ w > 0).astype(int)
    y_te = ((X_te - c) @ w > 0).astype(int)

    X_train.append(X_tr); y_train.append(y_tr)
    X_val.append(X_va);   y_val.append(y_va)
    X_test.append(X_te);  y_test.append(y_te)


# ═══════════════════════════════════════════════════
# EXPERT: MLP + density model + confidence output
# ═══════════════════════════════════════════════════

class Expert:
    """Isolated binary MLP. Provides:
       - predict(X) → class
       - confidence(X) → max softmax probability
       - density(X) → Gaussian density of input (fitted on training data)
    """
    def __init__(self, seed):
        rng = np.random.RandomState(seed)
        s = 0.1
        self.W1 = rng.randn(2, HIDDEN) * s
        self.b1 = np.zeros((1, HIDDEN))
        self.W2 = rng.randn(HIDDEN, 2) * s
        self.b2 = np.zeros((1, 2))
        # Density model (fitted after training)
        self.mean = None
        self.inv_cov = None
        self.det = None

    def forward(self, X):
        a = np.maximum(0, X @ self.W1 + self.b1)
        z = a @ self.W2 + self.b2
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def predict(self, X):
        return self.forward(X).argmax(1)

    def confidence(self, X):
        """Per-sample max softmax probability — 'how sure am I?'"""
        return self.forward(X).max(axis=1)

    def fit_density(self, X):
        """Fit Gaussian density model on training inputs."""
        self.mean = X.mean(axis=0)
        cov = np.cov(X.T)
        self.inv_cov = np.linalg.inv(cov + np.eye(2) * 1e-4)
        self.det = np.linalg.det(cov + np.eye(2) * 1e-4)

    def density(self, X):
        """Gaussian log-density (unnormalized) for input — higher = more typical."""
        d = X - self.mean
        return -0.5 * np.sum(d @ self.inv_cov * d, axis=1)

    def train(self, X, y, X_val=None, y_val=None, max_epochs=2000,
              early_stop=True, patience=50):
        """
        Train with early stopping on validation accuracy.
        Returns: best_val_acc, stopped_epoch
        """
        yo = np.eye(2)[y]
        best_acc = 0.0
        best_weights = None
        no_improve = 0
        stopped_epoch = max_epochs

        for ep in range(max_epochs):
            idx = np.random.choice(len(X), min(64, len(X)), replace=False)
            Xb, yb = X[idx], yo[idx]
            p = self.forward(Xb)
            N = len(Xb)
            a = np.maximum(0, Xb @ self.W1 + self.b1)
            dz = (p - yb) / N
            self.W2 -= LR * a.T @ dz
            self.b2 -= LR * dz.sum(0, keepdims=True)
            da = dz @ self.W2.T * (a > 0)
            self.W1 -= LR * Xb.T @ da
            self.b1 -= LR * da.sum(0, keepdims=True)

            if early_stop and X_val is not None and (ep + 1) % 10 == 0:
                val_acc = np.mean(self.predict(X_val) == y_val)
                if val_acc > best_acc + 0.001:
                    best_acc = val_acc
                    best_weights = (self.W1.copy(), self.b1.copy(),
                                    self.W2.copy(), self.b2.copy())
                    no_improve = 0
                else:
                    no_improve += 10
                if no_improve >= patience:
                    stopped_epoch = ep + 1
                    break

        # Restore best weights
        if best_weights is not None:
            self.W1, self.b1, self.W2, self.b2 = best_weights

        # Fit density model on training data
        self.fit_density(X)

        if X_val is not None:
            final_acc = np.mean(self.predict(X_val) == y_val)
        else:
            final_acc = 0.0
        return (best_acc if best_weights is not None else final_acc), stopped_epoch


class SGDNet:
    """SGD: shared W1, task-specific W2 columns."""
    def __init__(self):
        rng = np.random.RandomState(42)
        s = 0.1
        self.W1 = rng.randn(2, HIDDEN) * s
        self.b1 = np.zeros((1, HIDDEN))
        self.W2 = rng.randn(HIDDEN, N_TASKS * 2) * s
        self.b2 = np.zeros((1, N_TASKS * 2))

    def predict(self, X, tid):
        a = np.maximum(0, X @ self.W1 + self.b1)
        z = a @ self.W2 + self.b2
        e = np.exp(z - z.max(1, keepdims=True))
        return e[:, tid*2:tid*2+2].argmax(1)

    def train_task(self, X, y, tid, epochs=200):
        yo = np.eye(N_TASKS * 2)[y + tid * 2]
        for _ in range(epochs):
            idx = np.random.choice(len(X), min(64, len(X)), replace=False)
            Xb, yb = X[idx], yo[idx]
            a = np.maximum(0, Xb @ self.W1 + self.b1)
            z = a @ self.W2 + self.b2
            e = np.exp(z - z.max(1, keepdims=True))
            p = e / e.sum(1, keepdims=True)
            N = len(Xb)
            dz = (p - yb) / N
            self.W2 -= LR * a.T @ dz
            self.b2 -= LR * dz.sum(0, keepdims=True)
            da = dz @ self.W2.T * (a > 0)
            self.W1 -= LR * Xb.T @ da
            self.b1 -= LR * da.sum(0, keepdims=True)


# ═══════════════════════════════════════════════════
# ROUTERS (all use experts' own estimates — no separate module)
# ═══════════════════════════════════════════════════

def route_by_confidence(x, experts):
    """Each expert says how confident it is → pick the most confident.
    This IS the router-free principle: experts route themselves."""
    confs = [e.confidence(x.reshape(1, -1))[0] for e in experts]
    return np.argmax(confs)

def route_by_density(x, experts):
    """Each expert estimates input density → pick highest density.
    Also router-free: experts model their own input distribution."""
    dens = [e.density(x.reshape(1, -1))[0] for e in experts]
    return np.argmax(dens)

def route_by_prototype(x, prototypes):
    """Nearest cluster center. Not router-free — uses pre-computed prototypes.
    Included as baseline for comparison."""
    dists = [np.linalg.norm(x - p) for p in prototypes]
    return np.argmin(dists)


# ═══════════════════════════════════════════════════
# RUN: Epoch sweep × 5 seeds
# ═══════════════════════════════════════════════════

MAX_EPOCHS = 2000
EPOCH_STEPS = [50, 100, 200, 400, 800, 1600]
SEEDS = [42, 99, 123, 456, 789]

print("=" * 80)
print("END-TO-END ROUTING: Experts Route Themselves")
print("=" * 80)
print(f"Tasks: {N_TASKS} Gaussian clusters")
print(f"Routers: confidence (experts' own softmax) | density (experts' input model)")
print(f"         prototype (nearest mean — baseline)")
print(f"Expert training: max {MAX_EPOCHS} epochs, early stopping (patience=50)")
print(f"Seeds: {len(SEEDS)}  |  Epoch sweeps: {EPOCH_STEPS}")
print()

# Pre-compute prototypes
prototypes = [X_train[tid].mean(axis=0) for tid in range(N_TASKS)]

# Store per-epoch results for all routers
all_results = {}

for target_epochs in EPOCH_STEPS:
    print(f"\n{'─' * 80}")
    print(f"TARGET EPOCHS = {target_epochs}")
    print(f"{'─' * 80}")

    results = {
        'confidence': {'r_acc': [], 'e2e': []},
        'density':    {'r_acc': [], 'e2e': []},
        'prototype':  {'r_acc': [], 'e2e': []},
        'random':     {'r_acc': [], 'e2e': []},
        'perfect':    {'e2e': []},
        'sgd':        {'e2e': []},
    }

    for seed in SEEDS:
        # ── Train experts with early stopping ──
        experts = []
        stop_epochs = []
        for tid in range(N_TASKS):
            e = Expert(seed + tid)
            val_acc, stopped = e.train(
                X_train[tid], y_train[tid],
                X_val[tid], y_val[tid],
                max_epochs=target_epochs,
                early_stop=True, patience=50
            )
            experts.append(e)
            stop_epochs.append(stopped)

        # ── Train SGD ──
        sgd = SGDNet()
        for tid in range(N_TASKS):
            sgd.train_task(X_train[tid], y_train[tid], tid, epochs=target_epochs)

        # ── MIXED test set ──
        X_mixed = np.vstack(X_test)
        y_mixed = np.hstack(y_test)
        true_tid = np.hstack([np.full(N_TEST, tid) for tid in range(N_TASKS)])
        idx = np.random.RandomState(seed).permutation(len(X_mixed))
        X_mixed, y_mixed, true_tid = X_mixed[idx], y_mixed[idx], true_tid[idx]

        # ── Evaluate each router ──
        routers = [
            ('confidence', lambda x: route_by_confidence(x, experts)),
            ('density',    lambda x: route_by_density(x, experts)),
            ('prototype',  lambda x: route_by_prototype(x, prototypes)),
            ('random',     lambda x: np.random.RandomState(seed+999).randint(0, N_TASKS)),
        ]

        for rname, rfn in routers:
            routed = np.array([rfn(x) for x in X_mixed])
            r_acc = np.mean(routed == true_tid)
            preds = np.zeros(len(X_mixed), dtype=int)
            for tid in range(N_TASKS):
                mask = routed == tid
                if mask.sum() > 0:
                    preds[mask] = experts[tid].predict(X_mixed[mask])
            results[rname]['r_acc'].append(r_acc)
            results[rname]['e2e'].append(np.mean(preds == y_mixed))

        # ── Perfect routing ──
        perfect_preds = np.zeros(len(X_mixed), dtype=int)
        for tid in range(N_TASKS):
            mask = true_tid == tid
            if mask.sum() > 0:
                perfect_preds[mask] = experts[tid].predict(X_mixed[mask])
        results['perfect']['e2e'].append(np.mean(perfect_preds == y_mixed))

        # ── SGD ──
        sgd_preds = np.zeros(len(X_mixed), dtype=int)
        for tid in range(N_TASKS):
            mask = true_tid == tid
            if mask.sum() > 0:
                sgd_preds[mask] = sgd.predict(X_mixed[mask], tid)
        results['sgd']['e2e'].append(np.mean(sgd_preds == y_mixed))

    # ── Summary ──
    print(f"{'Router':<14} {'R_Acc':>8} {'R_±':>8} {'E2E':>8} {'E2E_±':>8} {'vs SGD':>8} {'vs Perf':>8}")
    print("-" * 66)

    sgd_mean = np.mean(results['sgd']['e2e'])
    perf_mean = np.mean(results['perfect']['e2e'])

    for rname in ['confidence', 'density', 'prototype', 'random']:
        ra = np.mean(results[rname]['r_acc'])
        rs = np.std(results[rname]['r_acc'])
        ea = np.mean(results[rname]['e2e'])
        es = np.std(results[rname]['e2e'])
        print(f"{rname:<14} {ra:>8.3f} {rs:>8.3f} {ea:>8.3f} {es:>8.3f} "
              f"{ea-sgd_mean:>+8.3f} {ea-perf_mean:>+8.3f}")

    print(f"{'perfect':<14} {'—':>8} {'—':>8} {perf_mean:>8.3f} "
          f"{np.std(results['perfect']['e2e']):>8.3f} {'—':>8} {'—':>8}")
    print(f"{'SGD':<14} {'—':>8} {'—':>8} {sgd_mean:>8.3f} "
          f"{np.std(results['sgd']['e2e']):>8.3f} {'—':>8} {'—':>8}")

    all_results[target_epochs] = results


# ═══════════════════════════════════════════════════
# FINAL: Full training curve with early stopping (seed=42)
# ═══════════════════════════════════════════════════
print(f"\n\n{'=' * 80}")
print("FULL TRAINING CURVE: Early stopping per expert (seed=42)")
print(f"{'=' * 80}")

seed = 42
# Train ONE expert per task with full tracking, NO early stop
# to show the overfitting curve AND where early stopping would cut

print(f"\n{'Epoch':<8} {'E0_val':>8} {'E0_test':>9} {'E1_val':>8} {'E1_test':>9} "
      f"{'E2_val':>8} {'E2_test':>9} {'E3_val':>8} {'E3_test':>9}")
print("-" * 82)

for target in [50, 100, 200, 400, 800, 1600]:
    experts_curve = []
    for tid in range(N_TASKS):
        e = Expert(seed + tid)
        # Train without early stop for the curve
        e.train(X_train[tid], y_train[tid], max_epochs=target,
                early_stop=False)
        val_acc = np.mean(e.predict(X_val[tid]) == y_val[tid])
        test_acc = np.mean(e.predict(X_test[tid]) == y_test[tid])
        experts_curve.append((val_acc, test_acc))

    print(f"{target:<8} " +
          f"{experts_curve[0][0]:>8.3f} {experts_curve[0][1]:>9.3f} " +
          f"{experts_curve[1][0]:>8.3f} {experts_curve[1][1]:>9.3f} " +
          f"{experts_curve[2][0]:>8.3f} {experts_curve[2][1]:>9.3f} " +
          f"{experts_curve[3][0]:>8.3f} {experts_curve[3][1]:>9.3f}")


# ═══════════════════════════════════════════════════
# DECOMPOSITION: best epoch for confidence router
# ═══════════════════════════════════════════════════
best_epoch = 400  # where PS peaks before overfitting
print(f"\n\n{'=' * 80}")
print(f"DETAILED DECOMPOSITION (epochs={best_epoch}, seed=42, confidence router)")
print(f"{'=' * 80}")

seed = 42
experts = []
for tid in range(N_TASKS):
    e = Expert(seed + tid)
    e.train(X_train[tid], y_train[tid], X_val[tid], y_val[tid],
            max_epochs=best_epoch, early_stop=True)
    experts.append(e)

X_mixed = np.vstack(X_test)
y_mixed = np.hstack(y_test)
true_tid = np.hstack([np.full(N_TEST, tid) for tid in range(N_TASKS)])

# Confidence router
routed = np.array([route_by_confidence(x, experts) for x in X_mixed])
r_acc = np.mean(routed == true_tid)

# Density router (comparison)
routed_d = np.array([route_by_density(x, experts) for x in X_mixed])
r_acc_d = np.mean(routed_d == true_tid)

print(f"\nConfidence router accuracy:  {r_acc:.3f}")
print(f"Density router accuracy:     {r_acc_d:.3f}")

# Confusion matrix for confidence router
print(f"\n{'Confusion matrix (true → routed) — CONFIDENCE:':>55}")
print(f"{'':>12}", end="")
for j in range(N_TASKS):
    print(f"{'→' + str(j):>8}", end="")
print()
for i in range(N_TASKS):
    print(f"{'Cluster ' + str(i):>12}", end="")
    for j in range(N_TASKS):
        count = np.sum((true_tid == i) & (routed == j))
        print(f"{count:>8}", end="")
    print()

# End-to-end
preds = np.zeros(len(X_mixed), dtype=int)
for tid in range(N_TASKS):
    mask = routed == tid
    if mask.sum() > 0:
        preds[mask] = experts[tid].predict(X_mixed[mask])
e2e = np.mean(preds == y_mixed)

# Decomposition
correct_mask = routed == true_tid
wrong_mask = routed != true_tid
correct_e2e = np.mean(preds[correct_mask] == y_mixed[correct_mask]) if correct_mask.sum() > 0 else 0
wrong_e2e = np.mean(preds[wrong_mask] == y_mixed[wrong_mask]) if wrong_mask.sum() > 0 else 0

# Perfect
perf_preds = np.zeros(len(X_mixed), dtype=int)
for tid in range(N_TASKS):
    mask = true_tid == tid
    if mask.sum() > 0:
        perf_preds[mask] = experts[tid].predict(X_mixed[mask])
perf_e2e = np.mean(perf_preds == y_mixed)

# SGD
sgd = SGDNet()
for tid in range(N_TASKS):
    sgd.train_task(X_train[tid], y_train[tid], tid, epochs=best_epoch)
sgd_preds = np.zeros(len(X_mixed), dtype=int)
for tid in range(N_TASKS):
    mask = true_tid == tid
    if mask.sum() > 0:
        sgd_preds[mask] = sgd.predict(X_mixed[mask], tid)
sgd_e2e = np.mean(sgd_preds == y_mixed)

print(f"\n{'DECOMPOSITION (confidence router):':>50}")
print(f"  Router accuracy:              {r_acc:.3f}")
print(f"  Expert acc (correct route):   {correct_e2e:.3f}")
print(f"  Expert acc (wrong route):     {wrong_e2e:.3f}")
print(f"  End-to-end (actual):          {e2e:.3f}")
print(f"  Perfect routing (ceiling):    {perf_e2e:.3f}")
print(f"  SGD (shared):                 {sgd_e2e:.3f}")
print(f"  Gap (PS − perfect):           {e2e - perf_e2e:+.3f}")
print(f"  Gap (PS − SGD):               {e2e - sgd_e2e:+.3f}")

# Expert quality check
print(f"\n{'Expert per-task accuracy (isolation quality):':>50}")
for tid in range(N_TASKS):
    mask = true_tid == tid
    e_acc = np.mean(experts[tid].predict(X_mixed[mask]) == y_mixed[mask])
    print(f"  Expert {tid}: {e_acc:.3f}")

print(f"\n{'=' * 80}")
print("EXPERIMENT COMPLETE — experts route themselves, no separate router")
print(f"{'=' * 80}")
