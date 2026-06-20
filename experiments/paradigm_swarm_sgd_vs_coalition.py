#!/usr/bin/env python3
"""SGD vs Coalition on compound queries — FAIR comparison.

Both see ONLY the 3 relevant tasks: colour, shape, size.
SGD: single 2-output head, sequential training.
PS: 3 isolated experts + emergent coalition.
2000 epochs per task. Compound test: 300 queries with all 3 signals strong.
"""
import numpy as np

DIM, HIDDEN, LR, BATCH = 20, 32, 0.005, 64
N_TRAIN, N_TEST = 300, 300
EPOCHS = 2000

np.random.seed(42)

# ═══════════════════════════════════════
# DATA — 3 tasks only
# ═══════════════════════════════════════
# Adversarial: colour + shape share dim1 with OPPOSITE signs
feat_c = [0, 1, 2];   sign_c = 1
feat_s = [1, 3, 5];   sign_s = -1   # adversarial on dim1
feat_z = [10, 11, 12]; sign_z = 1

def make_data(feats, sign, n=N_TRAIN):
    X = np.random.randn(n, DIM) * 0.3
    X[:, feats] = np.random.randn(n, len(feats)) * 3.0
    X += np.random.randn(*X.shape) * 0.2
    y = ((X[:, feats].sum(1) * sign) > 0).astype(int)
    return X, y

Xc_tr, yc_tr = make_data(feat_c, sign_c)
Xs_tr, ys_tr = make_data(feat_s, sign_s)
Xz_tr, yz_tr = make_data(feat_z, sign_z)

# Compound test: strong signal on all 3
all_f = list(set(feat_c + feat_s + feat_z))
X_comp = np.random.randn(N_TEST, DIM) * 0.3
for feats in [feat_c, feat_s, feat_z]:
    X_comp[:, feats] = np.random.randn(N_TEST, len(feats)) * 3.0
X_comp += np.random.randn(*X_comp.shape) * 0.2
yc_te = ((X_comp[:, feat_c].sum(1) * sign_c) > 0).astype(int)
ys_te = ((X_comp[:, feat_s].sum(1) * sign_s) > 0).astype(int)
yz_te = ((X_comp[:, feat_z].sum(1) * sign_z) > 0).astype(int)

# ═══════════════════════════════════════
# SGD — 3 separate output heads (fair: same as 3 isolated experts)
# ═══════════════════════════════════════
class SGDNet:
    def __init__(self):
        rng = np.random.RandomState(42); s = 0.03
        self.W1 = rng.randn(DIM, HIDDEN) * s; self.b1 = np.zeros((1, HIDDEN))
        self.W2c = rng.randn(HIDDEN, 2) * s; self.b2c = np.zeros((1, 2))
        self.W2s = rng.randn(HIDDEN, 2) * s; self.b2s = np.zeros((1, 2))
        self.W2z = rng.randn(HIDDEN, 2) * s; self.b2z = np.zeros((1, 2))
    def forward(self, X, head):
        a = np.maximum(0, X @ self.W1 + self.b1)
        if head == 'c': z = a @ self.W2c + self.b2c
        elif head == 's': z = a @ self.W2s + self.b2s
        else: z = a @ self.W2z + self.b2z
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
    def predict(self, X, head): return self.forward(X, head).argmax(1)
    def train_task(self, X, y, head, epochs):
        W2, b2 = {'c': (self.W2c, self.b2c), 's': (self.W2s, self.b2s),
                   'z': (self.W2z, self.b2z)}[head]
        for _ in range(epochs):
            idx = np.random.choice(len(X), BATCH, replace=False)
            Xb, yb = X[idx], y[idx]; N_b = len(Xb)
            a = np.maximum(0, Xb @ self.W1 + self.b1)
            z = a @ W2 + b2
            e = np.exp(z - z.max(1, keepdims=True)); e = e / e.sum(1, keepdims=True)
            yo = np.eye(2)[yb]; dz = (e - yo) / N_b
            W2 -= LR * a.T @ dz; b2 -= LR * dz.sum(0, keepdims=True)
            da = dz @ W2.T * (a > 0)
            self.W1 -= LR * Xb.T @ da; self.b1 -= LR * da.sum(0, keepdims=True)

# ═══════════════════════════════════════
# EXPERT + COALITION
# ═══════════════════════════════════════
class Expert:
    def __init__(self, seed):
        rng = np.random.RandomState(seed); s = 0.03
        self.W1 = rng.randn(DIM, HIDDEN) * s; self.b1 = np.zeros((1, HIDDEN))
        self.W2 = rng.randn(HIDDEN, 2) * s; self.b2 = np.zeros((1, 2)); self.tau = 0.5
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
    def calibrate(self, X, y, pct=20):
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

# ═══════════════════════════════════════
# EXPERIMENT
# ═══════════════════════════════════════
print("=" * 65)
print("SGD vs Coalition — 3 tasks ONLY, 2000 epochs, single 2-out head")
print(f"{N_TEST} TRUE compound queries")
print("=" * 65)

# ── SGD ──
sgd = SGDNet()
print("\n─── SGD sequential ───")
sgd.train_task(Xc_tr, yc_tr, 'c', EPOCHS)
c1 = np.mean(sgd.predict(X_comp, 'c') == yc_te)
print(f"  After colour:  c={c1:.3f}")

sgd.train_task(Xs_tr, ys_tr, 's', EPOCHS)
c2 = np.mean(sgd.predict(X_comp, 'c') == yc_te)
s2 = np.mean(sgd.predict(X_comp, 's') == ys_te)
print(f"  After shape:   c={c2:.3f} s={s2:.3f}")

sgd.train_task(Xz_tr, yz_tr, 'z', EPOCHS)
c3 = np.mean(sgd.predict(X_comp, 'c') == yc_te)
s3 = np.mean(sgd.predict(X_comp, 's') == ys_te)
z3 = np.mean(sgd.predict(X_comp, 'z') == yz_te)
print(f"  After size:    c={c3:.3f} s={s3:.3f} z={z3:.3f}  avg={np.mean([c3,s3,z3]):.3f}")

# ── PS ──
ce = Expert(301); ce.train(Xc_tr, yc_tr, EPOCHS)
se = Expert(302); se.train(Xs_tr, ys_tr, EPOCHS)
ze = Expert(303); ze.train(Xz_tr, yz_tr, EPOCHS)

pc = np.mean(ce.predict(X_comp) == yc_te)
ps = np.mean(se.predict(X_comp) == ys_te)
pz = np.mean(ze.predict(X_comp) == yz_te)
print(f"\n─── PS standalone ───")
print(f"  c={pc:.3f} s={ps:.3f} z={pz:.3f} avg={np.mean([pc,ps,pz]):.3f}")

# ── Coalition ──
cal_mix = np.vstack([Xc_tr[:100], Xs_tr[:100], Xz_tr[:100]])
cal_y0 = np.hstack([yc_tr[:100], np.zeros(100,int), np.zeros(100,int)])
cal_y1 = np.hstack([np.zeros(100,int), ys_tr[:100], np.zeros(100,int)])
cal_y2 = np.hstack([np.zeros(100,int), np.zeros(100,int), yz_tr[:100]])
ce.calibrate(cal_mix, cal_y0, pct=20)
se.calibrate(cal_mix, cal_y1, pct=20)
ze.calibrate(cal_mix, cal_y2, pct=20)

print(f"\n─── Coalition ───")
print(f"  Thresholds: c={ce.tau:.3f} s={se.tau:.3f} z={ze.tau:.3f}")

coal_c = {"colour":0,"shape":0,"size":0}
coal_t = {"colour":0,"shape":0,"size":0}
covered = all3 = 0; dims = [0,0,0,0]
for i in range(N_TEST):
    x = X_comp[i:i+1]
    coal = Coalition(x, [ce,se,ze], ["colour","shape","size"])
    res, _ = coal.resolve()
    if coal.size > 0: covered += 1
    dims[coal.size] += 1
    for d, truth in [("colour",yc_te[i]),("shape",ys_te[i]),("size",yz_te[i])]:
        if d in res: coal_t[d] += 1
        if d in res and res[d][0] == truth: coal_c[d] += 1
    if len(res) == 3:
        if res["colour"][0] == yc_te[i] and res["shape"][0] == ys_te[i] and res["size"][0] == yz_te[i]:
            all3 += 1

ca = {d: coal_c[d]/max(1,coal_t[d]) for d in ["colour","shape","size"]}
print(f"  Coverage: {covered/N_TEST:.2f}")
print(f"  Per-dim:  c={ca['colour']:.3f} s={ca['shape']:.3f} z={ca['size']:.3f}")
print(f"  Avg acc:  {np.mean(list(ca.values())):.3f}")
print(f"  All-3:    {all3}/{N_TEST} ({all3/N_TEST:.3f})")
print(f"  Dim dist: 0={dims[0]} 1={dims[1]} 2={dims[2]} 3={dims[3]}")

# ═══════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════
print(f"\n{'=' * 65}")
print("SUMMARY")
print(f"{'=' * 65}")
print(f"{'Metric':<20} {'SGD':>10} {'PS stand':>10} {'Coalition':>10}")
print("-" * 52)
print(f"{'Colour acc':<20} {c3:>10.3f} {pc:>10.3f} {ca['colour']:>10.3f}")
print(f"{'Shape acc':<20} {s3:>10.3f} {ps:>10.3f} {ca['shape']:>10.3f}")
print(f"{'Size acc':<20} {z3:>10.3f} {pz:>10.3f} {ca['size']:>10.3f}")
print(f"{'Avg':<20} {np.mean([c3,s3,z3]):>10.3f} {np.mean([pc,ps,pz]):>10.3f} {np.mean(list(ca.values())):>10.3f}")
print(f"{'Coverage':<20} {'100%':>10} {'100%':>10} {f'{covered/N_TEST:.0%}':>10}")
