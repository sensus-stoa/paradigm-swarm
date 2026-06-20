#!/usr/bin/env python3
"""50-Task Scaling with Density-Based Routing — End-to-End

Unlike §4.10 (oracle routing, identical X_base), this experiment gives each
task its OWN Gaussian cluster in 20D space. Input distributions are
distinguishable → density routing is possible → end-to-end PS vs SGD.

Key claims tested:
- Density routing accuracy on 50 well-separated clusters
- PS + density routing vs SGD at scale (40-320 epochs)
- Does the architectural advantage persist WITHOUT oracle routing?
"""
import numpy as np

np.random.seed(42)

# ═══════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════
N_TASKS = 50
DIM = 20
HIDDEN = 16
LR = 0.01
N_TRAIN = 200
N_TEST = 50
BATCH_SIZE = 32
SGD_EPOCHS = 40  # fixed for SGD (its optimal — longer → more interference)
PS_EPOCHS_LIST = [40, 80, 160, 320]
CLUSTER_RADIUS = 3.0  # distance of cluster centers from origin
CLUSTER_SIGMA = 0.5   # spread within cluster

# ═══════════════════════════════════════════════════
# GENERATE 50 DISTINGUISHABLE CLUSTERS
# ═══════════════════════════════════════════════════
# Place cluster centers on a 20D hypersphere
rng_centers = np.random.RandomState(123)
centers = rng_centers.randn(N_TASKS, DIM)
centers = centers / np.linalg.norm(centers, axis=1, keepdims=True) * CLUSTER_RADIUS

# Generate data per cluster + random classification hyperplane
X_train, y_train, X_test, y_test = [], [], [], []
for t in range(N_TASKS):
    rng_t = np.random.RandomState(1000 + t)
    # Data around cluster center
    X_tr = rng_t.randn(N_TRAIN, DIM) * CLUSTER_SIGMA + centers[t]
    X_te = rng_t.randn(N_TEST, DIM) * CLUSTER_SIGMA + centers[t]
    # Random hyperplane through the cluster center
    w = rng_t.randn(DIM)
    w = w / np.linalg.norm(w)
    sign = 1 if t % 2 == 0 else -1  # alternate for adversarial pressure on SGD
    y_tr = ((X_tr @ w) * sign > 0).astype(int)
    y_te = ((X_te @ w) * sign > 0).astype(int)
    X_train.append(X_tr); y_train.append(y_tr)
    X_test.append(X_te); y_test.append(y_te)

# ═══════════════════════════════════════════════════
# MLP EXPERT
# ═══════════════════════════════════════════════════
class Expert:
    def __init__(self, seed):
        rng = np.random.RandomState(seed)
        s = 0.05
        self.W1 = rng.randn(DIM, HIDDEN) * s
        self.b1 = np.zeros((1, HIDDEN))
        self.W2 = rng.randn(HIDDEN, 2) * s
        self.b2 = np.zeros((1, 2))
        # For density routing: fit Gaussian after training
        self.density_mean = None
        self.density_inv_cov = None
        self.density_log_det = None

    def forward(self, X):
        a = np.maximum(0, X @ self.W1 + self.b1)
        z = a @ self.W2 + self.b2
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def predict(self, X):
        return self.forward(X).argmax(1)

    def train(self, X, y, epochs):
        for _ in range(epochs):
            idx = np.random.choice(len(X), BATCH_SIZE, replace=False)
            Xb, yb = X[idx], y[idx]
            p = self.forward(Xb)
            N_b = len(Xb)
            a = np.maximum(0, Xb @ self.W1 + self.b1)
            yo = np.eye(2)[yb]
            dz = (p - yo) / N_b
            self.W2 -= LR * a.T @ dz
            self.b2 -= LR * dz.sum(0, keepdims=True)
            da = dz @ self.W2.T * (a > 0)
            self.W1 -= LR * Xb.T @ da
            self.b1 -= LR * da.sum(0, keepdims=True)

    def fit_density(self, X_train_data):
        """Fit Gaussian density model to training inputs for routing"""
        self.density_mean = X_train_data.mean(axis=0)
        cov = np.cov(X_train_data.T)
        # Regularize: add small diagonal for numerical stability
        cov += np.eye(DIM) * 1e-4
        self.density_inv_cov = np.linalg.inv(cov)
        sign, logdet = np.linalg.slogdet(cov)
        self.density_log_det = logdet

    def density(self, X):
        """Multivariate Gaussian log-density (unnormalized, for argmax)"""
        diff = X - self.density_mean
        mahal = np.sum(diff @ self.density_inv_cov * diff, axis=1)
        # Return negative Mahalanobis distance (proportional to log-density)
        return -0.5 * mahal - 0.5 * self.density_log_det

# ═══════════════════════════════════════════════════
# SGD: shared W1 + W2, sequential training
# ═══════════════════════════════════════════════════
class SGDNet:
    def __init__(self):
        rng = np.random.RandomState(42)
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

    def train_task(self, X, y, epochs):
        for _ in range(epochs):
            idx = np.random.choice(len(X), BATCH_SIZE, replace=False)
            Xb, yb = X[idx], y[idx]
            p = self.forward(Xb)
            N_b = len(Xb)
            a = np.maximum(0, Xb @ self.W1 + self.b1)
            yo = np.eye(2)[yb]
            dz = (p - yo) / N_b
            self.W2 -= LR * a.T @ dz
            self.b2 -= LR * dz.sum(0, keepdims=True)
            da = dz @ self.W2.T * (a > 0)
            self.W1 -= LR * Xb.T @ da
            self.b1 -= LR * da.sum(0, keepdims=True)

# ═══════════════════════════════════════════════════
# EXPERIMENT
# ═══════════════════════════════════════════════════
print("=" * 80)
print(f"50-TASK SCALING WITH DENSITY-BASED ROUTING — End-to-End")
print(f"{N_TASKS} tasks, {DIM}D, cluster_radius={CLUSTER_RADIUS}, sigma={CLUSTER_SIGMA}")
print(f"SGD: {SGD_EPOCHS} epochs/task. PS: {PS_EPOCHS_LIST} epochs/task")
print("=" * 80)

# --- SGD baseline ---
sgd_accs = {}
sgd = SGDNet()
for t in range(N_TASKS):
    sgd.train_task(X_train[t], y_train[t], SGD_EPOCHS)
# Test SGD on all tasks
sgd_per_task = [np.mean(sgd.predict(X_test[t]) == y_test[t]) for t in range(N_TASKS)]
sgd_avg = np.mean(sgd_per_task)
sgd_min = min(sgd_per_task)
sgd_harm = len(sgd_per_task) / sum(1.0/a for a in sgd_per_task)

print(f"\nSGD ({SGD_EPOCHS}ep/task): avg={sgd_avg:.3f}  harm={sgd_harm:.3f}  min={sgd_min:.3f}")

# --- Paradigm Swarm + Density Routing ---
print(f"\n{'PS ep':<8} {'R_Acc':<8} {'PS E2E':<8} {'PS min':<8} {'SGD':<8} {'Gap':<8} {'Winner':<8}")
print("-" * 64)

for ps_epochs in PS_EPOCHS_LIST:
    # Train isolated experts
    experts = []
    for t in range(N_TASKS):
        e = Expert(seed=1000 + t)
        e.train(X_train[t], y_train[t], ps_epochs)
        e.fit_density(X_train[t])
        experts.append(e)

    # End-to-end evaluation with density routing
    correct_route = 0
    correct_pred = 0
    total = 0
    ps_per_task = []

    # Mix all test queries into one pool (but track origin for routing accuracy)
    for t in range(N_TASKS):
        Xt = X_test[t]
        yt = y_test[t]
        n = len(Xt)

        # Density routing: compute density under ALL experts → argmax
        all_densities = np.column_stack([e.density(Xt) for e in experts])
        routed_to = all_densities.argmax(axis=1)

        # Predict using routed expert
        for i in range(n):
            pred = experts[routed_to[i]].predict(Xt[i:i+1])[0]
            if pred == yt[i]:
                correct_pred += 1
            if routed_to[i] == t:
                correct_route += 1
            total += 1

        # Per-task accuracy (via oracle routing — diagnostic only)
        oracle_pred = experts[t].predict(Xt)
        ps_per_task.append(np.mean(oracle_pred == yt))

    routing_acc = correct_route / total
    e2e_acc = correct_pred / total
    ps_avg = np.mean(ps_per_task)
    ps_min = min(ps_per_task)
    gap = e2e_acc - sgd_avg

    winner = "PS" if e2e_acc > sgd_avg else "SGD"
    print(f"{ps_epochs:<8} {routing_acc:<8.3f} {e2e_acc:<8.3f} {ps_min:<8.3f} "
          f"{sgd_avg:<8.3f} {gap:<-8.3f} {winner:<8}")

# ═══════════════════════════════════════════════════
# DETAIL: best PS configuration
# ═══════════════════════════════════════════════════
best_ep = PS_EPOCHS_LIST[-1]
experts = []
for t in range(N_TASKS):
    e = Expert(seed=1000 + t)
    e.train(X_train[t], y_train[t], best_ep)
    e.fit_density(X_train[t])
    experts.append(e)

# Per-task breakdown for best config
print(f"\n{'=' * 80}")
print(f"PER-TASK BREAKDOWN (PS epochs={best_ep})")
print(f"{'=' * 80}")

# Confusion matrix for density routing
confusion = np.zeros((N_TASKS, N_TASKS), dtype=int)
for t in range(N_TASKS):
    densities = np.column_stack([e.density(X_test[t]) for e in experts])
    routed = densities.argmax(axis=1)
    for r in routed:
        confusion[t, r] += 1

# Show routing accuracy per task
print(f"\nRouting accuracy per task (top-5 best / bottom-5 worst):")
routing_per_task = []
for t in range(N_TASKS):
    ra = confusion[t, t] / confusion[t].sum()
    routing_per_task.append((t, ra))

routing_per_task.sort(key=lambda x: -x[1])
print("  Best:  ", "  ".join(f"T{t}({ra:.2f})" for t, ra in routing_per_task[:5]))
print("  Worst: ", "  ".join(f"T{t}({ra:.2f})" for t, ra in routing_per_task[-5:]))

# E2E per task
e2e_per_task = []
for t in range(N_TASKS):
    Xt = X_test[t]; yt = y_test[t]
    densities = np.column_stack([e.density(Xt) for e in experts])
    routed_to = densities.argmax(axis=1)
    correct = sum(1 for i in range(len(Xt))
                  if experts[routed_to[i]].predict(Xt[i:i+1])[0] == yt[i])
    e2e_per_task.append(correct / len(Xt))

e2e_avg = np.mean(e2e_per_task)
e2e_min = min(e2e_per_task)

print(f"\nEnd-to-end per task: avg={e2e_avg:.3f}  min={e2e_min:.3f}")
print(f"  Best:  ", "  ".join(f"T{t}({e2e_per_task[t]:.2f})"
        for t in sorted(range(N_TASKS), key=lambda i: -e2e_per_task[i])[:5]))
print(f"  Worst: ", "  ".join(f"T{t}({e2e_per_task[t]:.2f})"
        for t in sorted(range(N_TASKS), key=lambda i: e2e_per_task[i])[:5]))

# Compare: oracle routing (upper bound) vs density routing
oracle_e2e = np.mean([np.mean(experts[t].predict(X_test[t]) == y_test[t])
                       for t in range(N_TASKS)])
print(f"\nOracle routing (upper bound): {oracle_e2e:.3f}")
print(f"Density routing (end-to-end):  {e2e_avg:.3f}")
print(f"Routing cost:                  {oracle_e2e - e2e_avg:.3f}")

# ═══════════════════════════════════════════════════
# MULTI-SEED
# ═══════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print(f"MULTI-SEED SUMMARY (5 seeds, PS={best_ep}ep, SGD={SGD_EPOCHS}ep)")
print(f"{'=' * 80}")
print(f"{'Seed':<8} {'R_Acc':<8} {'PS E2E':<8} {'SGD':<8} {'Gap':<8} {'Winner':<8}")
print("-" * 56)

multi_results = []
for seed in [42, 99, 123, 456, 789]:
    np.random.seed(seed)
    rng_c = np.random.RandomState(seed + 1000)
    ctrs = rng_c.randn(N_TASKS, DIM)
    ctrs = ctrs / np.linalg.norm(ctrs, axis=1, keepdims=True) * CLUSTER_RADIUS

    X_tr_m, y_tr_m, X_te_m, y_te_m = [], [], [], []
    for t in range(N_TASKS):
        rng_t = np.random.RandomState(seed * 100 + t)
        Xt = rng_t.randn(N_TRAIN, DIM) * CLUSTER_SIGMA + ctrs[t]
        Xv = rng_t.randn(N_TEST, DIM) * CLUSTER_SIGMA + ctrs[t]
        w = rng_t.randn(DIM); w = w / np.linalg.norm(w)
        sign = 1 if t % 2 == 0 else -1
        X_tr_m.append(Xt); y_tr_m.append(((Xt @ w) * sign > 0).astype(int))
        X_te_m.append(Xv); y_te_m.append(((Xv @ w) * sign > 0).astype(int))

    # SGD
    sgd_m = SGDNet()
    for t in range(N_TASKS):
        sgd_m.train_task(X_tr_m[t], y_tr_m[t], SGD_EPOCHS)
    sgd_m_avg = np.mean([np.mean(sgd_m.predict(X_te_m[t]) == y_te_m[t])
                          for t in range(N_TASKS)])

    # PS + density routing
    experts_m = []
    for t in range(N_TASKS):
        e = Expert(seed=seed * 100 + t)
        e.train(X_tr_m[t], y_tr_m[t], best_ep)
        e.fit_density(X_tr_m[t])
        experts_m.append(e)

    correct_r, correct_p, total_m = 0, 0, 0
    for t in range(N_TASKS):
        Xt = X_te_m[t]; yt = y_te_m[t]
        all_d = np.column_stack([e.density(Xt) for e in experts_m])
        routed = all_d.argmax(axis=1)
        for i in range(len(Xt)):
            if experts_m[routed[i]].predict(Xt[i:i+1])[0] == yt[i]:
                correct_p += 1
            if routed[i] == t:
                correct_r += 1
            total_m += 1

    r_acc = correct_r / total_m
    ps_e2e = correct_p / total_m
    gap_m = ps_e2e - sgd_m_avg
    winner_m = "PS" if ps_e2e > sgd_m_avg else "SGD"
    print(f"{seed:<8} {r_acc:<8.3f} {ps_e2e:<8.3f} {sgd_m_avg:<8.3f} "
          f"{gap_m:<8.3f} {winner_m:<8}")
    multi_results.append((r_acc, ps_e2e, sgd_m_avg, gap_m))

r_vals = [m[0] for m in multi_results]
ps_vals = [m[1] for m in multi_results]
sgd_vals = [m[2] for m in multi_results]
print("-" * 56)
print(f"{'Mean':<8} {np.mean(r_vals):<8.3f} {np.mean(ps_vals):<8.3f} "
      f"{np.mean(sgd_vals):<8.3f} {np.mean(ps_vals)-np.mean(sgd_vals):<8.3f}")
print(f"{'±Std':<8} {np.std(r_vals):<8.3f} {np.std(ps_vals):<8.3f} "
      f"{np.std(sgd_vals):<8.3f}")

print(f"\nDensity routing accuracy: {np.mean(r_vals):.3f} ± {np.std(r_vals):.3f}")
print(f"PS wins on {sum(1 for p,s in zip(ps_vals, sgd_vals) if p > s)}/{len(ps_vals)} seeds")
print(f"PS E2E: {np.mean(ps_vals):.3f} ± {np.std(ps_vals):.3f}")
print(f"SGD:    {np.mean(sgd_vals):.3f} ± {np.std(sgd_vals):.3f}")
print(f"Gap:    {np.mean(ps_vals)-np.mean(sgd_vals):.3f}")
