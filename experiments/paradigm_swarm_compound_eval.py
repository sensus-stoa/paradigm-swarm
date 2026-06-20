#!/usr/bin/env python3
"""Regenerate compound queries with ALL 3 dimensions strong,
then re-run coalition evaluation on trained experts."""

import numpy as np

DIM, HIDDEN, BATCH = 20, 32, 64
N_TASKS, N_TRAIN, N_TEST = 30, 300, 150
EPOCHS, LR = 2000, 0.005
seed = 42

np.random.seed(seed)
rng = np.random.RandomState(789)

# ── Same task generation as capital experiment ──
FEATURES = {}
for t in range(N_TASKS):
    nf = rng.randint(2, 5)
    feats = sorted(rng.choice(DIM, nf, replace=False).tolist())
    sign = 1 if t % 2 == 0 else -1
    FEATURES[t] = (feats, sign)
FEATURES[0] = ([0, 1, 2], 1)     # colour
FEATURES[1] = ([1, 3, 5], -1)    # shape
FEATURES[2] = ([10, 11, 12], 1)  # size

X_base = np.random.randn(max(N_TRAIN, N_TEST) * 2, DIM) * 0.5

X_train, y_train, X_test, y_test = [], [], [], []
for t in range(N_TASKS):
    feats, sign = FEATURES[t]
    X_tr = X_base[:N_TRAIN].copy(); X_tr[:, feats] *= 4.0
    X_tr += np.random.randn(*X_tr.shape) * 0.2
    X_te = X_base[N_TRAIN:N_TRAIN+N_TEST].copy(); X_te[:, feats] *= 4.0
    X_te += np.random.randn(*X_te.shape) * 0.2
    X_train.append(X_tr); y_train.append((X_tr[:, feats].sum(1) * sign > 0).astype(int))
    X_test.append(X_te); y_test.append((X_te[:, feats].sum(1) * sign > 0).astype(int))

# ── TRUE COMPOUND QUERIES: strong signal on ALL 3 dimension sets ──
# Activate colour feats [0,1,2], shape feats [1,3,5], size feats [10,11,12]
c_feats = [0, 1, 2]
s_feats = [1, 3, 5]
z_feats = [10, 11, 12]
all_feats = list(set(c_feats + s_feats + z_feats))

n_compound = 300
X_compound = np.random.randn(n_compound, DIM) * 0.3  # low base noise
# Amplify ALL three dimension sets
for feats in [c_feats, s_feats, z_feats]:
    X_compound[:, feats] = np.random.randn(n_compound, len(feats)) * 3.0
X_compound += np.random.randn(*X_compound.shape) * 0.2

# Ground truth for compound queries
c_sign = FEATURES[0][1]; s_sign = FEATURES[1][1]; z_sign = FEATURES[2][1]
yc = ((X_compound[:, c_feats].sum(1) * c_sign) > 0).astype(int)
ys = ((X_compound[:, s_feats].sum(1) * s_sign) > 0).astype(int)
yz = ((X_compound[:, z_feats].sum(1) * z_sign) > 0).astype(int)

print(f"Compound queries generated: {n_compound}")
print(f"  Colour signal strength:  mean|X[:,0]|={np.abs(X_compound[:,0]).mean():.2f}")
print(f"  Shape signal strength:   mean|X[:,1]|={np.abs(X_compound[:,1]).mean():.2f}")
print(f"  Size signal strength:    mean|X[:,10]|={np.abs(X_compound[:,10]).mean():.2f}")
print(f"  Colour baseline: {yc.mean():.2f}  Shape baseline: {ys.mean():.2f}  Size baseline: {yz.mean():.2f}")

# ── Train experts (same as capital experiment) ──
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

    def calibrate(self, X, y, pct=10):
        confs = np.array([self.confidence(x.reshape(1, -1)) for x in X])
        self.tau = np.percentile(confs, 100 - pct)

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
        return result, "bound" if not conflicts else f"bound+{len(conflicts)}"

# Train all 30 experts
experts = []
for t in range(N_TASKS):
    e = Expert(1000 + t)
    e.train(X_train[t], y_train[t], EPOCHS)
    experts.append(e)

# Standalone accuracy on compound queries
ps_c = np.mean(experts[0].predict(X_compound) == yc)
ps_s = np.mean(experts[1].predict(X_compound) == ys)
ps_z = np.mean(experts[2].predict(X_compound) == yz)
print(f"\nPS standalone on compound: c={ps_c:.3f} s={ps_s:.3f} z={ps_z:.3f} avg={np.mean([ps_c,ps_s,ps_z]):.3f}")

# ── Coalition with mixed calibration ──
cal_mix = np.vstack([X_train[0][:100], X_train[1][:100], X_train[2][:100]])
cal_y0 = np.hstack([y_train[0][:100], np.zeros(100,int), np.zeros(100,int)])
cal_y1 = np.hstack([np.zeros(100,int), y_train[1][:100], np.zeros(100,int)])
cal_y2 = np.hstack([np.zeros(100,int), np.zeros(100,int), y_train[2][:100]])
experts[0].calibrate(cal_mix, cal_y0, pct=30)
experts[1].calibrate(cal_mix, cal_y1, pct=30)
experts[2].calibrate(cal_mix, cal_y2, pct=30)

print(f"\nThresholds: colour={experts[0].tau:.3f} shape={experts[1].tau:.3f} size={experts[2].tau:.3f}")

# ── Coalition evaluation ──
coal_c = {"colour":0,"shape":0,"size":0}
coal_t = {"colour":0,"shape":0,"size":0}
covered = 0; all3 = 0; dims = [0,0,0,0]  # 0,1,2,3 dims covered

for i in range(n_compound):
    x = X_compound[i:i+1]
    coal = Coalition(x, experts[:3], ["colour","shape","size"])
    res, _ = coal.resolve()
    if coal.size > 0: covered += 1
    dims[coal.size] += 1
    for d, truth in [("colour",yc[i]),("shape",ys[i]),("size",yz[i])]:
        if d in res: coal_t[d] += 1
        if d in res and res[d][0] == truth: coal_c[d] += 1
    if len(res) == 3:
        if res["colour"][0] == yc[i] and res["shape"][0] == ys[i] and res["size"][0] == yz[i]:
            all3 += 1

coal_acc = {d: coal_c[d]/max(1, coal_t[d]) for d in ["colour","shape","size"]}
coal_avg = np.mean(list(coal_acc.values()))
coal_cov = covered / n_compound

print(f"\nCoalition results on {n_compound} TRUE compound queries:")
print(f"  Coverage: {coal_cov:.2f} ({covered}/{n_compound})")
print(f"  Per-dim acc: c={coal_acc['colour']:.3f} s={coal_acc['shape']:.3f} z={coal_acc['size']:.3f}")
print(f"  Avg accuracy: {coal_avg:.3f}")
print(f"  All-3-correct: {all3}/{n_compound} ({all3/n_compound:.3f})")
print(f"  Dim distribution: 0={dims[0]} 1={dims[1]} 2={dims[2]} 3={dims[3]}")
