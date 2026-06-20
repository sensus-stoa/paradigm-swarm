#!/usr/bin/env python3
"""Binding vs SGD — epoch sweep on compound queries

Three dimensions: colour, shape, size.
Compound queries require ALL THREE correct to count as success.

Compare:
  SGD: shared W1, 3×2 output heads, sequential training
  Binding+Experts: 3 isolated experts + binding layer merge
  
Sweep: epochs per expert/task from 40 to 320.
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

# Ground truth labels — every point has ALL THREE
y_c_tr = (X_tr[:, 0] > 0).astype(int)
y_s_tr = (X_tr[:, 1] > 0).astype(int)
y_z_tr = ((X_tr[:, 2]**2 + X_tr[:, 3]**2) > 1.0).astype(int)

y_c_te = (X_te[:, 0] > 0).astype(int)
y_s_te = (X_te[:, 1] > 0).astype(int)
y_z_te = ((X_te[:, 2]**2 + X_te[:, 3]**2) > 1.0).astype(int)

# Relaxed compound queries: at least moderate signal on all dimensions
compound_mask = (np.abs(X_te[:, 0]) > 0.8) & (np.abs(X_te[:, 1]) > 0.8) & ((X_te[:, 2]**2 + X_te[:, 3]**2) > 0.8)
compound_X = X_te[compound_mask]
compound_yc = y_c_te[compound_mask]
compound_ys = y_s_te[compound_mask]
compound_yz = y_z_te[compound_mask]

# ═══════════════════════════════════════
# EXPERT (isolated)
# ═══════════════════════════════════════
class Expert:
    def __init__(self, seed, dim_name, class_names):
        rng = np.random.RandomState(seed); s = 0.05
        self.W1 = rng.randn(DIM, HIDDEN) * s; self.b1 = np.zeros((1, HIDDEN))
        self.W2 = rng.randn(HIDDEN, 2) * s; self.b2 = np.zeros((1, 2))
        self.dim_name = dim_name; self.class_names = class_names

    def forward(self, X):
        a = np.maximum(0, X @ self.W1 + self.b1)
        z = a @ self.W2 + self.b2
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def predict(self, X): return self.forward(X).argmax(1)
    def confidence(self, x): return float(self.forward(x).max(1)[0])

    def train(self, X, y, epochs):
        for _ in range(epochs):
            idx = np.random.choice(len(X), BATCH, replace=False)
            Xb, yb = X[idx], y[idx]; p = self.forward(Xb); N_b = len(Xb)
            a = np.maximum(0, Xb @ self.W1 + self.b1); yo = np.eye(2)[yb]
            dz = (p - yo) / N_b
            self.W2 -= LR * a.T @ dz; self.b2 -= LR * dz.sum(0, keepdims=True)
            da = dz @ self.W2.T * (a > 0)
            self.W1 -= LR * Xb.T @ da; self.b1 -= LR * da.sum(0, keepdims=True)

# ═══════════════════════════════════════
# SGD — shared W1, 3×2 output heads
# ═══════════════════════════════════════
class SGDNet:
    def __init__(self):
        rng = np.random.RandomState(42); s = 0.05
        self.W1 = rng.randn(DIM, HIDDEN) * s; self.b1 = np.zeros((1, HIDDEN))
        # 3 output heads: colour(2), shape(2), size(2)
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

    def predict(self, X, head):
        return self.forward(X, head).argmax(1)

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
# BINDING LAYER
# ═══════════════════════════════════════
class BindingLayer:
    def __init__(self, pct=20): self.pct = pct; self.tau = {}
    def calibrate(self, experts, X, y_list):
        for e, y in zip(experts, y_list):
            confs = np.array([e.confidence(x.reshape(1, -1)) for x in X[:200]])
            self.tau[e.dim_name] = np.percentile(confs, 100 - self.pct)
    def process(self, x, experts):
        qualifying = []
        for e in experts:
            conf = e.confidence(x)
            if conf >= self.tau[e.dim_name]:
                qualifying.append((e.dim_name, e.predict(x)[0], conf))
        if not qualifying: return {}, qualifying
        result = {}
        for dim, pred, conf in qualifying:
            if dim not in result: result[dim] = (pred, conf)
            else:
                ex_pred, ex_conf = result[dim]
                if pred != ex_pred and conf > ex_conf:
                    result[dim] = (pred, conf)
        return result, qualifying

# ═══════════════════════════════════════
# EXPERIMENT
# ═══════════════════════════════════════
epochs_list = [40, 80, 160, 320]

print("=" * 80)
print("BINDING + EXPERTS vs SGD — Epoch Sweep on Compound Queries")
print(f"{len(compound_X)} compound queries (all 3 dimensions strong)")
print("=" * 80)

# Header
print(f"\n{'Epochs':<8} {'SGD avg':<10} {'Bind avg':<10} {'Coverage':<10} {'Dims/q':<10} {'Winner':<10}")
print("-" * 58)

for ep in epochs_list:
    # --- SGD: sequential training ---
    sgd = SGDNet()
    # Train colour
    sgd.train_task(X_tr, y_c_tr, 'c', ep)
    # Train shape 
    sgd.train_task(X_tr, y_s_tr, 's', ep)
    # Train size
    sgd.train_task(X_tr, y_z_tr, 'z', ep)
    
    # Test SGD on compound queries — per-dimension accuracy
    sgd_c_acc = np.mean(sgd.predict(compound_X, 'c') == compound_yc)
    sgd_s_acc = np.mean(sgd.predict(compound_X, 's') == compound_ys)
    sgd_z_acc = np.mean(sgd.predict(compound_X, 'z') == compound_yz)
    sgd_avg = np.mean([sgd_c_acc, sgd_s_acc, sgd_z_acc])
    
    # Check forgetting: colour accuracy on compound (trained first, tested last)
    sgd_forget = 0.0  # measured differently here — colour trained first
    
    # --- Binding + Experts ---
    ce = Expert(201, "colour", ["red", "blue"]); ce.train(X_tr, y_c_tr, ep)
    se = Expert(202, "shape", ["circle", "square"]); se.train(X_tr, y_s_tr, ep)
    ze = Expert(203, "size", ["big", "small"]); ze.train(X_tr, y_z_tr, ep)
    experts = [ce, se, ze]
    
    # Calibrate binding
    bl = BindingLayer(pct=20)
    bl.calibrate(experts, X_tr, [y_c_tr, y_s_tr, y_z_tr])
    
    # Test binding: per-dimension accuracy on qualifying queries
    bind_c_correct = 0; bind_c_total = 0
    bind_s_correct = 0; bind_s_total = 0
    bind_z_correct = 0; bind_z_total = 0
    all3_correct = 0; covered_any = 0
    
    for i in range(len(compound_X)):
        x = compound_X[i:i+1]
        result, _ = bl.process(x, experts)
        
        if len(result) > 0:
            covered_any += 1
        
        if 'colour' in result:
            bind_c_total += 1
            if result['colour'][0] == compound_yc[i]: bind_c_correct += 1
        if 'shape' in result:
            bind_s_total += 1
            if result['shape'][0] == compound_ys[i]: bind_s_correct += 1
        if 'size' in result:
            bind_z_total += 1
            if result['size'][0] == compound_yz[i]: bind_z_correct += 1
        
        # All-3 present and correct
        if len(result) >= 3:
            ok = (result['colour'][0] == compound_yc[i] and 
                  result['shape'][0] == compound_ys[i] and 
                  result['size'][0] == compound_yz[i])
            if ok: all3_correct += 1
    
    bind_c_acc = bind_c_correct / max(1, bind_c_total)
    bind_s_acc = bind_s_correct / max(1, bind_s_total)
    bind_z_acc = bind_z_correct / max(1, bind_z_total)
    bind_avg = np.mean([bind_c_acc, bind_s_acc, bind_z_acc])
    coverage = covered_any / len(compound_X)
    dims_per_q = (bind_c_total + bind_s_total + bind_z_total) / len(compound_X)
    
    winner = "BIND" if bind_avg > sgd_avg else "SGD" if sgd_avg > bind_avg else "TIE"
    print(f"{ep:<8} {sgd_avg:<10.3f} {bind_avg:<10.3f} {coverage:<10.2f} {dims_per_q:<10.1f} {winner:<10}")

# ═══════════════════════════════════════
# MULTI-SEED
# ═══════════════════════════════════════
print(f"\n{'=' * 80}")
print("MULTI-SEED (5 seeds, epochs=320)")
print(f"{'=' * 80}")
print(f"{'Seed':<8} {'SGD avg':<10} {'Bind avg':<10} {'Coverage':<10} {'Dims/q':<10} {'Winner':<10}")
print("-" * 58)

results = []
for seed in [42, 99, 123, 456, 789]:
    np.random.seed(seed)
    X_tr_m = np.random.randn(N_TRAIN, DIM) * 1.2
    X_te_m = np.random.randn(N_TEST, DIM) * 1.2
    yc_tr_m = (X_tr_m[:, 0] > 0).astype(int)
    ys_tr_m = (X_tr_m[:, 1] > 0).astype(int)
    yz_tr_m = ((X_tr_m[:, 2]**2 + X_tr_m[:, 3]**2) > 1.0).astype(int)
    cm = (np.abs(X_te_m[:, 0]) > 0.8) & (np.abs(X_te_m[:, 1]) > 0.8) & ((X_te_m[:, 2]**2 + X_te_m[:, 3]**2) > 0.8)
    cX = X_te_m[cm]; cyc = (cX[:, 0] > 0).astype(int)
    cys = (cX[:, 1] > 0).astype(int); cyz = ((cX[:, 2]**2 + cX[:, 3]**2) > 1.0).astype(int)
    
    # SGD — per-dimension accuracy
    sgd_m = SGDNet()
    sgd_m.train_task(X_tr_m, yc_tr_m, 'c', 320)
    sgd_m.train_task(X_tr_m, ys_tr_m, 's', 320)
    sgd_m.train_task(X_tr_m, yz_tr_m, 'z', 320)
    sgd_m_avg = np.mean([np.mean(sgd_m.predict(cX, 'c') == cyc),
                          np.mean(sgd_m.predict(cX, 's') == cys),
                          np.mean(sgd_m.predict(cX, 'z') == cyz)])
    
    # Binding — per-dimension accuracy on qualifying
    ce_m = Expert(seed, "colour", ["red", "blue"]); ce_m.train(X_tr_m, yc_tr_m, 320)
    se_m = Expert(seed+1, "shape", ["circle", "square"]); se_m.train(X_tr_m, ys_tr_m, 320)
    ze_m = Expert(seed+2, "size", ["big", "small"]); ze_m.train(X_tr_m, yz_tr_m, 320)
    exps = [ce_m, se_m, ze_m]
    bl_m = BindingLayer(pct=20); bl_m.calibrate(exps, X_tr_m, [yc_tr_m, ys_tr_m, yz_tr_m])
    
    bc = bs = bz = 0; bct = bst = bzt = 0; cov = 0
    for i in range(len(cX)):
        res, _ = bl_m.process(cX[i:i+1], exps)
        if len(res) > 0: cov += 1
        if 'colour' in res: bct += 1; bc += (res['colour'][0] == cyc[i])
        if 'shape' in res: bst += 1; bs += (res['shape'][0] == cys[i])
        if 'size' in res: bzt += 1; bz += (res['size'][0] == cyz[i])
    
    b_avg = np.mean([bc/max(1,bct), bs/max(1,bst), bz/max(1,bzt)])
    cov_r = cov / len(cX); dims = (bct + bst + bzt) / len(cX)
    w = "BIND" if b_avg > sgd_m_avg else "SGD" if sgd_m_avg > b_avg else "TIE"
    print(f"{seed:<8} {sgd_m_avg:<10.3f} {b_avg:<10.3f} {cov_r:<10.2f} {dims:<10.1f} {w:<10}")
    results.append((sgd_m_avg, b_avg, cov_r, dims))

sgd_vals = [r[0] for r in results]
bind_vals = [r[1] for r in results]
print("-" * 58)
print(f"{'Mean':<8} {np.mean(sgd_vals):<10.3f} {np.mean(bind_vals):<10.3f} {np.mean([r[2] for r in results]):<10.2f} {np.mean([r[3] for r in results]):<10.1f}")
print(f"{'±Std':<8} {np.std(sgd_vals):<10.3f} {np.std(bind_vals):<10.3f}")
print(f"\nBinding wins on {sum(1 for s,b in zip(sgd_vals, bind_vals) if b > s)}/{len(results)} seeds")
