#!/usr/bin/env python3
"""Emergent Binding — experts form temporary coalitions, no stored layer

Key insight: binding is NOT a separate component. When multiple experts
qualify for a query, they self-organize into a temporary coalition,
apply merge rules, return the result, and dissolve.

Compare: Coalition (emergent) vs SGD (shared backbone) vs Argmax (naive).
"""
import numpy as np

np.random.seed(42)
DIM, HIDDEN, LR, BATCH = 4, 12, 0.01, 32
N_TRAIN, N_TEST = 400, 200

# ═══════════════════════════════════════
# DATA
# ═══════════════════════════════════════
X_tr = np.random.randn(N_TRAIN, DIM) * 1.2
X_te = np.random.randn(N_TEST, DIM) * 1.2

y_c = (X_tr[:, 0] > 0).astype(int); y_c_te = (X_te[:, 0] > 0).astype(int)
y_s = (X_tr[:, 1] > 0).astype(int); y_s_te = (X_te[:, 1] > 0).astype(int)
y_z = ((X_tr[:, 2]**2 + X_tr[:, 3]**2) > 1.0).astype(int)
y_z_te = ((X_te[:, 2]**2 + X_te[:, 3]**2) > 1.0).astype(int)

cm = (np.abs(X_te[:, 0]) > 0.8) & (np.abs(X_te[:, 1]) > 0.8) & ((X_te[:, 2]**2 + X_te[:, 3]**2) > 0.8)
cX = X_te[cm]; cyc = y_c_te[cm]; cys = y_s_te[cm]; cyz = y_z_te[cm]

# ═══════════════════════════════════════
# EXPERT — self-aware, can join coalitions
# ═══════════════════════════════════════
class Expert:
    def __init__(self, seed, dim_name, class_names):
        rng = np.random.RandomState(seed); s = 0.05
        self.W1 = rng.randn(DIM, HIDDEN) * s; self.b1 = np.zeros((1, HIDDEN))
        self.W2 = rng.randn(HIDDEN, 2) * s; self.b2 = np.zeros((1, 2))
        self.dim_name = dim_name; self.class_names = class_names
        self.tau = None  # calibrated threshold

    def forward(self, X):
        a = np.maximum(0, X @ self.W1 + self.b1)
        z = a @ self.W2 + self.b2
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def predict(self, X): return self.forward(X).argmax(1)
    def predict_label(self, x): return self.class_names[self.predict(x)[0]]
    def confidence(self, x): return float(self.forward(x).max(1)[0])
    def qualifies(self, x): return self.confidence(x) >= (self.tau or 0.5)

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
        confs = np.array([self.confidence(x.reshape(1, -1)) for x in X[:200]])
        self.tau = np.percentile(confs, 100 - pct)

# ═══════════════════════════════════════
# COALITION — temporary, per-query, self-destructs
# ═══════════════════════════════════════
class Coalition:
    """A temporary assembly of experts for ONE query. 
    
    Created when router finds multiple qualifying experts.
    Applies merge rules, returns result, then discarded.
    NOT stored anywhere. NOT a layer. A transient event.
    """
    def __init__(self, query, experts):
        self.members = []
        for e in experts:
            if e.qualifies(query):
                pred_idx = e.predict(query)[0]  # 0 or 1
                conf = e.confidence(query)
                self.members.append((e.dim_name, pred_idx, conf))
    
    @property
    def size(self): return len(self.members)
    
    def resolve(self):
        """Apply 4 structural rules and return merged result."""
        if not self.members:
            return {}, "GAP"
        
        if len(self.members) == 1:
            dim, label, conf = self.members[0]
            return {dim: (label, conf)}, "single"
        
        result = {}; conflicts = []
        for dim, label, conf in self.members:
            if dim not in result:
                result[dim] = (label, conf)              # R1: merge
            else:
                ex_label, ex_conf = result[dim]
                if label == ex_label:
                    result[dim] = (label, max(conf, ex_conf))  # R2: reinforce
                else:
                    conflicts.append((dim, ex_label, label))
                    if conf > ex_conf:
                        result[dim] = (label, conf)       # R3: strongest wins
        
        verdict = "bound" if not conflicts else f"bound [+{len(conflicts)}]"
        return result, verdict

# ═══════════════════════════════════════
# SGD
# ═══════════════════════════════════════
class SGDNet:
    def __init__(self):
        rng = np.random.RandomState(42); s = 0.05
        self.W1 = rng.randn(DIM, HIDDEN) * s; self.b1 = np.zeros((1, HIDDEN))
        self.W2c = rng.randn(HIDDEN, 2) * s; self.b2c = np.zeros((1, 2))
        self.W2s = rng.randn(HIDDEN, 2) * s; self.b2s = np.zeros((1, 2))
        self.W2z = rng.randn(HIDDEN, 2) * s; self.b2z = np.zeros((1, 2))

    def fwd(self, X, head):
        a = np.maximum(0, X @ self.W1 + self.b1)
        if head == 'c': z = a @ self.W2c + self.b2c
        elif head == 's': z = a @ self.W2s + self.b2s
        else: z = a @ self.W2z + self.b2z
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def pred(self, X, head): return self.fwd(X, head).argmax(1)

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
# TRAIN & CALIBRATE
# ═══════════════════════════════════════
def make_experts(ep):
    ce = Expert(201, "colour", ["red", "blue"]); ce.train(X_tr, y_c, ep)
    se = Expert(202, "shape", ["circle", "square"]); se.train(X_tr, y_s, ep)
    ze = Expert(203, "size", ["big", "small"]); ze.train(X_tr, y_z, ep)
    ce.calibrate(X_tr, y_c); se.calibrate(X_tr, y_s); ze.calibrate(X_tr, y_z)
    return [ce, se, ze]

# ═══════════════════════════════════════
# EXPERIMENT
# ═══════════════════════════════════════
epochs_list = [40, 80, 160, 320]

print("=" * 75)
print("EMERGENT BINDING — Coalition vs SGD vs Argmax")
print(f"{len(cX)} compound queries")
print("=" * 75)

print(f"\n{'Ep':<6} {'SGD avg':<10} {'Coal avg':<10} {'Argmax avg':<10} "
      f"{'Coal cov':<10} {'Coal size':<10} {'Winner':<10}")
print("-" * 66)

for ep in epochs_list:
    # --- SGD ---
    sgd = SGDNet()
    sgd.train_task(X_tr, y_c, 'c', ep)
    sgd.train_task(X_tr, y_s, 's', ep)
    sgd.train_task(X_tr, y_z, 'z', ep)
    sgd_avg = np.mean([np.mean(sgd.pred(cX, 'c') == cyc),
                        np.mean(sgd.pred(cX, 's') == cys),
                        np.mean(sgd.pred(cX, 'z') == cyz)])
    
    # --- Experts ---
    experts = make_experts(ep)
    
    # --- Coalition (emergent binding) ---
    coal_correct = {"colour": 0, "shape": 0, "size": 0}
    coal_total   = {"colour": 0, "shape": 0, "size": 0}
    coal_covered = 0; total_members = 0
    
    # --- Argmax (naive) ---
    argmax_correct = {"colour": 0, "shape": 0, "size": 0}
    
    for i in range(len(cX)):
        x = cX[i:i+1]
        
        # Coalition: form temporary assembly, resolve, discard
        coal = Coalition(x, experts)
        result, _ = coal.resolve()
        # Coalition object goes out of scope here → garbage collected
        
        if coal.size > 0:
            coal_covered += 1
            total_members += coal.size
        
        for dim in ["colour", "shape", "size"]:
            if dim in result:
                coal_total[dim] += 1
                if dim == "colour" and result[dim][0] == cyc[i]: coal_correct[dim] += 1
                if dim == "shape" and result[dim][0] == cys[i]: coal_correct[dim] += 1
                if dim == "size" and result[dim][0] == cyz[i]: coal_correct[dim] += 1
        
        # Argmax baseline
        confs = [e.confidence(x) for e in experts]
        best = np.argmax(confs)
        for j, dim in enumerate(["colour", "shape", "size"]):
            pred = experts[j].predict(x)[0]
            if j == best:  # only count the argmax winner's dimension
                truth = [cyc[i], cys[i], cyz[i]][j]
                if pred == truth: argmax_correct[dim] += 1
    
    coal_acc = [coal_correct[d]/max(1, coal_total[d]) for d in ["colour","shape","size"]]
    coal_avg = np.mean(coal_acc)
    coal_cov = coal_covered / len(cX)
    coal_sz  = total_members / len(cX)
    
    arg_acc = [argmax_correct[d]/len(cX) for d in ["colour","shape","size"]]
    arg_avg = np.mean(arg_acc)
    
    winner = "COAL" if coal_avg > sgd_avg else "SGD" if sgd_avg > coal_avg else "TIE"
    print(f"{ep:<6} {sgd_avg:<10.3f} {coal_avg:<10.3f} {arg_avg:<10.3f} "
          f"{coal_cov:<10.2f} {coal_sz:<10.1f} {winner:<10}")

# ═══════════════════════════════════════
# MULTI-SEED
# ═══════════════════════════════════════
print(f"\n{'=' * 75}")
print("MULTI-SEED (5 seeds, epochs=320)")
print(f"{'=' * 75}")
print(f"{'Seed':<8} {'SGD avg':<10} {'Coal avg':<10} {'Coal cov':<10} {'Coal sz':<10} {'Winner':<10}")
print("-" * 58)

results = []
for seed in [42, 99, 123, 456, 789]:
    np.random.seed(seed)
    Xm = np.random.randn(N_TRAIN, DIM) * 1.2; Xt = np.random.randn(N_TEST, DIM) * 1.2
    ycm = (Xm[:, 0] > 0).astype(int); ysm = (Xm[:, 1] > 0).astype(int)
    yzm = ((Xm[:, 2]**2 + Xm[:, 3]**2) > 1.0).astype(int)
    ccm = (np.abs(Xt[:, 0]) > 0.8) & (np.abs(Xt[:, 1]) > 0.8) & ((Xt[:, 2]**2 + Xt[:, 3]**2) > 0.8)
    cXm = Xt[ccm]; cycm = (cXm[:, 0] > 0).astype(int)
    cysm = (cXm[:, 1] > 0).astype(int); cyzm = ((cXm[:, 2]**2 + cXm[:, 3]**2) > 1.0).astype(int)
    
    sgd_m = SGDNet()
    sgd_m.train_task(Xm, ycm, 'c', 320)
    sgd_m.train_task(Xm, ysm, 's', 320)
    sgd_m.train_task(Xm, yzm, 'z', 320)
    sgd_m_avg = np.mean([np.mean(sgd_m.pred(cXm, 'c') == cycm),
                          np.mean(sgd_m.pred(cXm, 's') == cysm),
                          np.mean(sgd_m.pred(cXm, 'z') == cyzm)])
    
    cem = Expert(seed, "colour", ["red", "blue"]); cem.train(Xm, ycm, 320)
    sem = Expert(seed+1, "shape", ["circle", "square"]); sem.train(Xm, ysm, 320)
    zem = Expert(seed+2, "size", ["big", "small"]); zem.train(Xm, yzm, 320)
    cem.calibrate(Xm, ycm); sem.calibrate(Xm, ysm); zem.calibrate(Xm, yzm)
    exps = [cem, sem, zem]
    
    cc = {"colour":0,"shape":0,"size":0}; ct = {"colour":0,"shape":0,"size":0}
    cov = 0; tmem = 0
    for i in range(len(cXm)):
        x = cXm[i:i+1]; coal = Coalition(x, exps); res, _ = coal.resolve()
        if coal.size > 0: cov += 1; tmem += coal.size
        for d in ["colour","shape","size"]:
            if d in res:
                ct[d] += 1
                truth = [cycm[i], cysm[i], cyzm[i]][["colour","shape","size"].index(d)]
                if res[d][0] == truth: cc[d] += 1
    
    ca = np.mean([cc[d]/max(1,ct[d]) for d in ["colour","shape","size"]])
    cr = cov/len(cXm); cs = tmem/len(cXm)
    w = "COAL" if ca > sgd_m_avg else "SGD" if sgd_m_avg > ca else "TIE"
    print(f"{seed:<8} {sgd_m_avg:<10.3f} {ca:<10.3f} {cr:<10.2f} {cs:<10.1f} {w:<10}")
    results.append((sgd_m_avg, ca, cr, cs))

sv = [r[0] for r in results]; cv = [r[1] for r in results]
print("-" * 58)
print(f"{'Mean':<8} {np.mean(sv):<10.3f} {np.mean(cv):<10.3f} "
      f"{np.mean([r[2] for r in results]):<10.2f} {np.mean([r[3] for r in results]):<10.1f}")
print(f"{'±Std':<8} {np.std(sv):<10.3f} {np.std(cv):<10.3f}")
print(f"Coalition wins on {sum(1 for s,c in zip(sv,cv) if c > s)}/{len(results)} seeds")
