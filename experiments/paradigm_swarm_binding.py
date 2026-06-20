#!/usr/bin/env python3
"""Binding Layer Experiment — calibrated confidence routing + structural merge

Three experts: colour, shape, size. Each trained on FULL input space.
Confidence routing: per-expert threshold = 95th percentile of confidence 
on training data. Experts above threshold qualify for binding.

Binding layer: 4 structural rules, zero parameters.
Compare: argmax (loses dimensions) vs binding (preserves all).
"""
import numpy as np

np.random.seed(42)
DIM, HIDDEN, LR, N_TRAIN, N_TEST = 4, 12, 0.01, 400, 100

# ═══════════════════════════════════════
# DATA: every point has all three labels
# ═══════════════════════════════════════
X_train = np.random.randn(N_TRAIN, DIM) * 1.2
X_test  = np.random.randn(N_TEST,  DIM) * 1.2

y_colour = (X_train[:, 0] > 0).astype(int)
y_shape  = (X_train[:, 1] > 0).astype(int)
y_size   = ((X_train[:, 2]**2 + X_train[:, 3]**2) > 1.0).astype(int)

y_colour_te = (X_test[:, 0] > 0).astype(int)
y_shape_te  = (X_test[:, 1] > 0).astype(int)
y_size_te   = ((X_test[:, 2]**2 + X_test[:, 3]**2) > 1.0).astype(int)

# ═══════════════════════════════════════
# EXPERT
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
    def predict_label(self, x): return self.class_names[self.predict(x)[0]]
    def confidence(self, x): return float(self.forward(x).max(1)[0])

    def train(self, X, y, epochs=400):
        for _ in range(epochs):
            idx = np.random.choice(len(X), 32, replace=False)
            Xb, yb = X[idx], y[idx]; p = self.forward(Xb); N_b = len(Xb)
            a = np.maximum(0, Xb @ self.W1 + self.b1); yo = np.eye(2)[yb]
            dz = (p - yo) / N_b
            self.W2 -= LR * a.T @ dz; self.b2 -= LR * dz.sum(0, keepdims=True)
            da = dz @ self.W2.T * (a > 0)
            self.W1 -= LR * Xb.T @ da; self.b1 -= LR * da.sum(0, keepdims=True)

# ═══════════════════════════════════════
# TRAIN
# ═══════════════════════════════════════
colour_e = Expert(201, "colour", ["red", "blue"]); colour_e.train(X_train, y_colour)
shape_e  = Expert(202, "shape",  ["circle", "square"]); shape_e.train(X_train, y_shape)
size_e   = Expert(203, "size",   ["big", "small"]); size_e.train(X_train, y_size)
experts = [colour_e, shape_e, size_e]

acc_c = np.mean(colour_e.predict(X_test) == y_colour_te)
acc_s = np.mean(shape_e.predict(X_test) == y_shape_te)
acc_z = np.mean(size_e.predict(X_test) == y_size_te)

# ═══════════════════════════════════════
# BINDING LAYER — confidence routing + structural merge
# ═══════════════════════════════════════
class BindingLayer:
    """Confidence routing → structural merge. Zero semantic knowledge.
    
    Calibration: per-expert threshold = 90th percentile of confidence 
    on training data. Expert qualifies if its confidence exceeds its 
    own calibrated threshold.
    
    Merge rules:
      R1: dimensions different → merge all
      R2: same dim, same value → reinforce
      R3: same dim, different value → conflict (strongest wins)
      R4: single qualifying expert → return as-is
    """
    def __init__(self, pct=10):
        self.pct = pct  # lower pct = stricter (pct=10 → 90th percentile)
        self.tau = {}
    
    def calibrate(self, experts, X_list):
        for e, X in zip(experts, X_list):
            confs = np.array([e.confidence(x.reshape(1, -1)) for x in X[:200]])
            self.tau[e.dim_name] = np.percentile(confs, 100 - self.pct)
    
    def process(self, x, experts):
        qualifying = []
        for e in experts:
            conf = e.confidence(x)
            if conf >= self.tau[e.dim_name]:
                label = e.predict_label(x)
                qualifying.append((e.dim_name, label, conf))
        
        if not qualifying:
            return {"_verdict": "GAP"}, qualifying
        
        if len(qualifying) == 1:
            dim, label, conf = qualifying[0]
            return {dim: (label, conf), "_verdict": "single"}, qualifying
        
        result = {}; conflicts = []
        for dim, label, conf in qualifying:
            if dim not in result:
                result[dim] = (label, conf)              # R1
            else:
                ex_label, ex_conf = result[dim]
                if label == ex_label:
                    result[dim] = (label, max(conf, ex_conf))  # R2
                else:
                    conflicts.append((dim, ex_label, label))
                    if conf > ex_conf:
                        result[dim] = (label, conf)       # R3
        
        v = "bound" if not conflicts else f"bound [+{len(conflicts)} conflict]"
        result["_verdict"] = v
        return result, qualifying

binding = BindingLayer(pct=20)
binding.calibrate(experts, [X_train, X_train, X_train])

# ═══════════════════════════════════════
# TEST
# ═══════════════════════════════════════
queries = {
    "big red circle":     np.array([ 2.0,  2.0,  1.5,  0.5]),
    "big blue square":    np.array([-2.0, -2.0,  1.5,  0.5]),
    "small red square":   np.array([ 2.0, -2.0,  0.3,  0.3]),
    "small blue circle":  np.array([-2.0,  2.0,  0.3,  0.3]),
    "red":     np.array([ 2.0,  0.0,  0.0,  0.0]),
    "circle":  np.array([ 0.0,  2.0,  0.0,  0.0]),
    "big":     np.array([ 0.0,  0.0,  1.5,  1.0]),
    "muddy":   np.array([ 0.1, -0.1,  0.1,  0.1]),
}

print("=" * 65)
print("BINDING LAYER — calibrated confidence + structural merge")
print("=" * 65)
print(f"Accuracy: colour={acc_c:.3f} shape={acc_s:.3f} size={acc_z:.3f}")
print(f"Threshold: top {100-binding.pct}% training confidence per expert")
for e in experts:
    print(f"  {e.dim_name}: tau={binding.tau[e.dim_name]:.3f}")
print()
print(f"{'Query':<22} {'Argmax':<16} {'Binding':<38} {'Qualifying':>18}")
print("-" * 96)

compound_names = ["big red circle", "big blue square", "small red square", "small blue circle"]
single_names   = ["red", "circle", "big"]

for name, q in queries.items():
    x = q.reshape(1, -1)
    
    # Argmax
    confs = [e.confidence(x) for e in experts]
    best = np.argmax(confs)
    argmax_str = f"{experts[best].dim_name}:{experts[best].predict_label(x)}"
    
    # Binding
    result, qualifying = binding.process(x, experts)
    verdict = result.pop("_verdict", "?")
    parts = [f"{dim}:{label}({conf:.2f})" for dim, (label, conf) in result.items()]
    binding_str = " + ".join(parts) if parts else "GAP"
    if "conflict" in verdict: binding_str += " [!]"
    
    qual_str = ", ".join(f"{dim}" for dim, _, _ in qualifying) if qualifying else "none"
    print(f"{name:<22} {argmax_str:<16} {binding_str:<38} {qual_str:>18}")

# ═══════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════
print(f"\n{'=' * 65}")
print("SUMMARY")
print(f"{'=' * 65}")

argmax_dims = 0; binding_dims = 0
for name in compound_names:
    x = queries[name].reshape(1, -1)
    argmax_dims += 1
    result, _ = binding.process(x, experts)
    result.pop("_verdict", None)
    binding_dims += len(result)

total = len(compound_names) * 3
print(f"Compound queries ({len(compound_names)}):")
print(f"  Argmax  preserved: {argmax_dims}/{total} ({100*argmax_dims/total:.0f}%)")
print(f"  Binding preserved: {binding_dims}/{total} ({100*binding_dims/total:.0f}%)")

fp = 0
for name in single_names:
    x = queries[name].reshape(1, -1)
    result, _ = binding.process(x, experts)
    result.pop("_verdict", None)
    if len(result) > 1: fp += 1
print(f"  False multi-binding on single-dim: {fp}/{len(single_names)}")

# GAP
r_m, q_m = binding.process(queries["muddy"].reshape(1, -1), experts)
r_m.pop("_verdict", None)
print(f"  Ambiguous 'muddy': {len(r_m)} dims ({len(q_m)} qualifying)")
