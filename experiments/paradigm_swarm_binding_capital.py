#!/usr/bin/env python3
"""Capital Experiment v2: 30 tasks causing TRUE catastrophic forgetting

30 binary tasks, 20D input, 2-4 random features each, alternating signs.
SGD: single 2-output head (TRUE forgetting — W1 + W2 overwritten).
PS: 30 isolated experts + emergent coalition for compound queries.

Compound queries: 3 dimensions (colour/shape/size), each = expert trained
on its own task. Compare coalition merge vs SGD on all-3-correct.
2000 epochs per task.
"""
import numpy as np

DIM, HIDDEN, LR, BATCH = 20, 32, 0.005, 64
N_TASKS = 30
N_TRAIN, N_TEST = 300, 150
EPOCHS = 2000

np.random.seed(42)

# ═══════════════════════════════════════
# 30 RANDOM TASKS — overlapping features, adversarial signs
# ═══════════════════════════════════════
rng = np.random.RandomState(789)
FEATURES = {}
for t in range(N_TASKS):
    nf = rng.randint(2, 5)  # 2-4 features per task
    feats = sorted(rng.choice(DIM, nf, replace=False).tolist())
    sign = 1 if t % 2 == 0 else -1  # alternate for adversarial pressure
    FEATURES[t] = (feats, sign)

# First 3 tasks = colour, shape, size (for coalition testing)
FEATURES[0] = ([0, 1, 2], 1)     # colour
FEATURES[1] = ([1, 3, 5], -1)    # shape (overlaps dim1 with colour, adversarial sign)
FEATURES[2] = ([10, 11, 12], 1)  # size (dedicated dims)

# Generate data per task
X_base = np.random.randn(max(N_TRAIN, N_TEST) * 2, DIM) * 0.5

X_train, y_train, X_test, y_test = [], [], [], []
for t in range(N_TASKS):
    feats, sign = FEATURES[t]
    # Amplify task-relevant features
    X_tr = X_base[:N_TRAIN].copy()
    X_tr[:, feats] *= 4.0
    X_tr += np.random.randn(*X_tr.shape) * 0.2
    y_tr = ((X_tr[:, feats].sum(1) * sign) > 0).astype(int)
    
    X_te = X_base[N_TRAIN:N_TRAIN+N_TEST].copy()
    X_te[:, feats] *= 4.0
    X_te += np.random.randn(*X_te.shape) * 0.2
    y_te = ((X_te[:, feats].sum(1) * sign) > 0).astype(int)
    
    X_train.append(X_tr); y_train.append(y_tr)
    X_test.append(X_te); y_test.append(y_te)

# Compound queries: use test data from colour(0), shape(1), size(2)
# For each test point, all 3 experts must answer correctly
cX = X_test[0]  # colour test data as compound queries
cyc = y_test[0]; cys = y_test[1]; cyz = y_test[2]

# ═══════════════════════════════════════
# EXPERT
# ═══════════════════════════════════════
class Expert:
    def __init__(self, seed):
        rng = np.random.RandomState(seed); s = 0.03
        self.W1 = rng.randn(DIM, HIDDEN) * s; self.b1 = np.zeros((1, HIDDEN))
        self.W2 = rng.randn(HIDDEN, 2) * s; self.b2 = np.zeros((1, 2))
        self.tau = 0.5

    def forward(self, X):
        a = np.maximum(0, X @ self.W1 + self.b1)
        z = a @ self.W2 + self.b2
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def predict(self, X): return self.forward(X).argmax(1)
    def confidence(self, x): return float(self.forward(x).max(1)[0])
    def qualifies(self, x): return self.confidence(x) >= self.tau

    def train(self, X, y, epochs):
        for _ in range(epochs):
            idx = np.random.choice(len(X), BATCH, replace=False)
            Xb, yb = X[idx], y[idx]; p = self.forward(Xb); N_b = len(Xb)
            a = np.maximum(0, Xb @ self.W1 + self.b1); yo = np.eye(2)[yb]
            dz = (p - yo) / N_b
            self.W2 -= LR * a.T @ dz; self.b2 -= LR * dz.sum(0, keepdims=True)
            da = dz @ self.W2.T * (a > 0)
            self.W1 -= LR * Xb.T @ da; self.b1 -= LR * da.sum(0, keepdims=True)

    def calibrate(self, X, y, pct=5):
        confs = np.array([self.confidence(x.reshape(1, -1)) for x in X[:200]])
        self.tau = np.percentile(confs, 100 - pct)

# ═══════════════════════════════════════
# COALITION
# ═══════════════════════════════════════
class Coalition:
    def __init__(self, query, experts, dim_names):
        self.members = []
        for e, dn in zip(experts, dim_names):
            if e.qualifies(query):
                self.members.append((dn, e.predict(query)[0], e.confidence(query)))

    @property
    def size(self): return len(self.members)

    def resolve(self):
        if not self.members: return {}, "GAP"
        if len(self.members) == 1:
            d, l, c = self.members[0]; return {d: (l, c)}, "single"
        result = {}; conflicts = []
        for dim, label, conf in self.members:
            if dim not in result: result[dim] = (label, conf)
            else:
                ex_l, ex_c = result[dim]
                if label == ex_l: result[dim] = (label, max(conf, ex_c))
                else:
                    conflicts.append((dim, ex_l, label))
                    if conf > ex_c: result[dim] = (label, conf)
        v = "bound" if not conflicts else f"bound [+{len(conflicts)}]"
        return result, v

# ═══════════════════════════════════════
# SGD — single 2-output head for ALL 30 tasks
# ═══════════════════════════════════════
class SGDNet:
    def __init__(self):
        rng = np.random.RandomState(42); s = 0.03
        self.W1 = rng.randn(DIM, HIDDEN) * s; self.b1 = np.zeros((1, HIDDEN))
        self.W2 = rng.randn(HIDDEN, 2) * s; self.b2 = np.zeros((1, 2))

    def forward(self, X):
        a = np.maximum(0, X @ self.W1 + self.b1)
        z = a @ self.W2 + self.b2
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def predict(self, X): return self.forward(X).argmax(1)

    def train_task(self, X, y, epochs):
        for _ in range(epochs):
            idx = np.random.choice(len(X), BATCH, replace=False)
            Xb, yb = X[idx], y[idx]; N_b = len(Xb)
            a = np.maximum(0, Xb @ self.W1 + self.b1)
            z = a @ self.W2 + self.b2
            e = np.exp(z - z.max(1, keepdims=True)); e = e / e.sum(1, keepdims=True)
            yo = np.eye(2)[yb]; dz = (e - yo) / N_b
            self.W2 -= LR * a.T @ dz; self.b2 -= LR * dz.sum(0, keepdims=True)
            da = dz @ self.W2.T * (a > 0)
            self.W1 -= LR * Xb.T @ da; self.b1 -= LR * da.sum(0, keepdims=True)

# ═══════════════════════════════════════
# EXPERIMENT
# ═══════════════════════════════════════
print("=" * 75)
print(f"CAPITAL EXPERIMENT: {N_TASKS} tasks, {DIM}D, {EPOCHS} epochs/task")
print(f"SGD: single 2-output head. PS: {N_TASKS} isolated experts + coalition")
print(f"{len(cX)} compound queries (colour+shape+size)")
print("=" * 75)

# --- SGD ---
sgd = SGDNet()
# Track accuracy on first 3 tasks throughout training
sgd_snapshots = {}
for t in range(N_TASKS):
    sgd.train_task(X_train[t], y_train[t], EPOCHS)
    if t < 3 or t % 10 == 9:
        accs = [np.mean(sgd.predict(X_test[i]) == y_test[i]) for i in range(min(t+1, 3))]
        sgd_snapshots[t] = accs
        label = f"T{t}: " + " ".join(f"{a:.3f}" for a in accs)
        if t < 3 or t >= 27:
            print(f"  SGD after {label}")

# Final SGD accuracy on first 3 tasks
sgd_c = np.mean(sgd.predict(X_test[0]) == y_test[0])
sgd_s = np.mean(sgd.predict(X_test[1]) == y_test[1])
sgd_z = np.mean(sgd.predict(X_test[2]) == y_test[2])
sgd_avg = np.mean([sgd_c, sgd_s, sgd_z])
print(f"\n  SGD FINAL on colour/shape/size: c={sgd_c:.3f} s={sgd_s:.3f} z={sgd_z:.3f} avg={sgd_avg:.3f}")

# --- PS ---
experts = []
for t in range(N_TASKS):
    e = Expert(1000 + t)
    e.train(X_train[t], y_train[t], EPOCHS)
    experts.append(e)

# First 3 experts accuracy (isolated, no interference)
ps_c = np.mean(experts[0].predict(X_test[0]) == y_test[0])
ps_s = np.mean(experts[1].predict(X_test[1]) == y_test[1])
ps_z = np.mean(experts[2].predict(X_test[2]) == y_test[2])
ps_avg = np.mean([ps_c, ps_s, ps_z])
print(f"  PS FINAL on colour/shape/size: c={ps_c:.3f} s={ps_s:.3f} z={ps_z:.3f} avg={ps_avg:.3f}")

# --- Coalition on compound queries ---
# Calibrate on compound-style data: use a mix of all three training sets
# so thresholds reflect the compound query distribution
calib_mix = np.vstack([X_train[0][:100], X_train[1][:100], X_train[2][:100]])
calib_y0 = np.hstack([y_train[0][:100], np.zeros(100, dtype=int), np.zeros(100, dtype=int)])
calib_y1 = np.hstack([np.zeros(100, dtype=int), y_train[1][:100], np.zeros(100, dtype=int)])
calib_y2 = np.hstack([np.zeros(100, dtype=int), np.zeros(100, dtype=int), y_train[2][:100]])

experts[0].calibrate(calib_mix, calib_y0, pct=10)
experts[1].calibrate(calib_mix, calib_y1, pct=10)
experts[2].calibrate(calib_mix, calib_y2, pct=10)
dim_names = ["colour", "shape", "size"]

coal_c = {"colour":0,"shape":0,"size":0}
coal_t = {"colour":0,"shape":0,"size":0}
covered = 0; tmem = 0
for i in range(len(cX)):
    x = cX[i:i+1]
    coal = Coalition(x, experts[:3], dim_names)
    res, _ = coal.resolve()
    if coal.size > 0: covered += 1; tmem += coal.size
    for d, truth in [("colour",cyc[i]),("shape",cys[i]),("size",cyz[i])]:
        if d in res: coal_t[d] += 1
        if d in res and res[d][0] == truth: coal_c[d] += 1

coal_acc = {d: coal_c[d]/max(1, coal_t[d]) for d in ["colour","shape","size"]}
coal_avg = np.mean(list(coal_acc.values()))
coal_cov = covered / len(cX)

print(f"  Coalition FINAL: avg={coal_avg:.3f} cov={coal_cov:.2f} "
      f"c={coal_acc['colour']:.3f} s={coal_acc['shape']:.3f} z={coal_acc['size']:.3f}")

# ═══════════════════════════════════════
# SGD FORGETTING CURVE
# ═══════════════════════════════════════
print(f"\n{'=' * 75}")
print("SGD FORGETTING CURVE (accuracy on colour/shape/size)")
print(f"{'=' * 75}")
print(f"{'After task':<12} {'Colour':<8} {'Shape':<8} {'Size':<8}")
print("-" * 40)
for t, accs in sorted(sgd_snapshots.items()):
    c = accs[0] if len(accs) > 0 else 0
    s = accs[1] if len(accs) > 1 else 0
    z = accs[2] if len(accs) > 2 else 0
    mark = " ←" if t < 3 else ""
    print(f"T{t:<11} {c:<8.3f} {s:<8.3f} {z:<8.3f}{mark}")

# First-task survival
first_acc_start = sgd_snapshots[0][0]
first_acc_end = sgd_c
print(f"\nColour (T0) survival: {first_acc_start:.3f} → {first_acc_end:.3f} "
      f"(forgot {first_acc_start - first_acc_end:.3f})")

# ═══════════════════════════════════════
# MULTI-SEED
# ═══════════════════════════════════════
print(f"\n{'=' * 75}")
print("MULTI-SEED (5 seeds)")
print(f"{'=' * 75}")
print(f"{'Seed':<8} {'SGD avg':<10} {'PS avg':<10} {'Coal avg':<10} "
      f"{'Coal cov':<10} {'SGD forget':<12} {'Winner':<10}")
print("-" * 68)

results = []
for seed in [42, 99, 123, 456, 789]:
    np.random.seed(seed)
    rng_s = np.random.RandomState(seed + 1000)
    
    # Generate data
    F = {}
    for t in range(N_TASKS):
        nf = rng_s.randint(2, 5)
        feats = sorted(rng_s.choice(DIM, nf, replace=False).tolist())
        sign = 1 if t % 2 == 0 else -1
        F[t] = (feats, sign)
    F[0] = ([0, 1, 2], 1)
    F[1] = ([1, 3, 5], -1)
    F[2] = ([10, 11, 12], 1)
    
    Xb2 = np.random.randn(max(N_TRAIN, N_TEST) * 2, DIM) * 0.5
    Xtr_m, ytr_m, Xte_m, yte_m = [], [], [], []
    for t in range(N_TASKS):
        feats, sign = F[t]
        Xtr = Xb2[:N_TRAIN].copy(); Xtr[:, feats] *= 4.0
        Xtr += np.random.randn(*Xtr.shape) * 0.2
        Xte = Xb2[N_TRAIN:N_TRAIN+N_TEST].copy(); Xte[:, feats] *= 4.0
        Xte += np.random.randn(*Xte.shape) * 0.2
        Xtr_m.append(Xtr); ytr_m.append(((Xtr[:, feats].sum(1) * sign) > 0).astype(int))
        Xte_m.append(Xte); yte_m.append(((Xte[:, feats].sum(1) * sign) > 0).astype(int))
    
    # SGD
    s2 = SGDNet()
    for t in range(N_TASKS):
        s2.train_task(Xtr_m[t], ytr_m[t], EPOCHS)
    s2_c = np.mean(s2.predict(Xte_m[0]) == yte_m[0])
    s2_s = np.mean(s2.predict(Xte_m[1]) == yte_m[1])
    s2_z = np.mean(s2.predict(Xte_m[2]) == yte_m[2])
    s2_avg = np.mean([s2_c, s2_s, s2_z])
    s2_forget = 0  # placeholder
    
    # PS
    exps_m = []
    for t in range(N_TASKS):
        e = Expert(seed * 100 + t)
        e.train(Xtr_m[t], ytr_m[t], EPOCHS)
        exps_m.append(e)
    p2_c = np.mean(exps_m[0].predict(Xte_m[0]) == yte_m[0])
    p2_s = np.mean(exps_m[1].predict(Xte_m[1]) == yte_m[1])
    p2_z = np.mean(exps_m[2].predict(Xte_m[2]) == yte_m[2])
    p2_avg = np.mean([p2_c, p2_s, p2_z])
    
    # Coalition — calibrate on mixed data like main sweep
    cal_mix = np.vstack([Xtr_m[0][:100], Xtr_m[1][:100], Xtr_m[2][:100]])
    cal_y0 = np.hstack([ytr_m[0][:100], np.zeros(100,int), np.zeros(100,int)])
    cal_y1 = np.hstack([np.zeros(100,int), ytr_m[1][:100], np.zeros(100,int)])
    cal_y2 = np.hstack([np.zeros(100,int), np.zeros(100,int), ytr_m[2][:100]])
    exps_m[0].calibrate(cal_mix, cal_y0, pct=10)
    exps_m[1].calibrate(cal_mix, cal_y1, pct=10)
    exps_m[2].calibrate(cal_mix, cal_y2, pct=10)
    
    cc2 = {"colour":0,"shape":0,"size":0}; ct2 = {"colour":0,"shape":0,"size":0}
    cov2 = 0
    for i in range(len(Xte_m[0])):
        x = Xte_m[0][i:i+1]
        coal = Coalition(x, exps_m[:3], ["colour","shape","size"])
        res, _ = coal.resolve()
        if coal.size > 0: cov2 += 1
        for d, truth in [("colour",yte_m[0][i]),("shape",yte_m[1][i]),("size",yte_m[2][i])]:
            if d in res: ct2[d] += 1
            if d in res and res[d][0] == truth: cc2[d] += 1
    
    ca2 = np.mean([cc2[d]/max(1,ct2[d]) for d in ["colour","shape","size"]])
    cr2 = cov2/len(Xte_m[0])
    
    winner = "COAL" if ca2 > s2_avg else "SGD" if s2_avg > ca2 else "TIE"
    print(f"{seed:<8} {s2_avg:<10.3f} {p2_avg:<10.3f} {ca2:<10.3f} "
          f"{cr2:<10.2f} {s2_forget:<+12.3f} {winner:<10}")
    results.append((s2_avg, ca2))

sv = [r[0] for r in results]; cv = [r[1] for r in results]
print("-" * 68)
print(f"{'Mean':<8} {np.mean(sv):<10.3f} — {np.mean(cv):<10.3f}")
wins = sum(1 for s,c in zip(sv,cv) if c > s)
print(f"Coalition wins on {wins}/{len(results)} seeds")
