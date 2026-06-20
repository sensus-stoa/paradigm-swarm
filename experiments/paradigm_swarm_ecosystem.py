#!/usr/bin/env python3
"""Paradigm Ecosystem: known intersections, mutual improvement"""
import numpy as np

np.random.seed(42)
DIM, HIDDEN, LR = 60, 32, 0.03

N_TRAIN = [40, 300, 200, 80, 150]
FEATURES = {
    0: list(range(0,10)) + list(range(30,40)),     # shape + pattern
    1: list(range(0,10)) + list(range(10,20)),      # shape + texture
    2: list(range(10,20)) + list(range(20,30)),     # texture + color
    3: list(range(0,10)) + list(range(20,30)),      # shape + color
    4: list(range(30,40)) + list(range(40,50)),     # pattern + edge
}

# Precompute intersections
def intersect(a, b):
    return sorted(set(FEATURES[a]) & set(FEATURES[b]))

INTERSECTIONS = {}
for i in range(5):
    for j in range(i+1, 5):
        shared = intersect(i, j)
        if shared:
            INTERSECTIONS[(i,j)] = shared

X_base = np.random.randn(1500, DIM) * 0.3
datasets = []
for eid in range(5):
    X = X_base.copy(); X[:, FEATURES[eid]] *= 6
    y = (X[:, FEATURES[eid]].sum(1) > 0).astype(int)
    n = N_TRAIN[eid]
    datasets.append((X[:n], y[:n], X[500:700], y[500:700]))

class Expert:
    def __init__(self, eid):
        rng=np.random.RandomState(42+eid); s=0.05
        self.W1=rng.randn(DIM,HIDDEN)*s; self.b1=np.zeros((1,HIDDEN))
        self.W2=rng.randn(HIDDEN,2)*s; self.b2=np.zeros((1,2)); self.eid=eid
    def fwd(self,X):
        a=np.maximum(0,X@self.W1+self.b1); z=a@self.W2+self.b2
        e=np.exp(z-z.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)
    def pred(self,X): return self.fwd(X).argmax(1)
    def train(self,X,y,epochs=300,freeze_W1=False):
        for _ in range(epochs):
            p=self.fwd(X); N=len(X); a=np.maximum(0,X@self.W1+self.b1)
            yo=np.eye(2)[y]; dz=(p-yo)/N
            self.W2-=LR*a.T@dz; self.b2-=LR*dz.sum(0,keepdims=True)
            if not freeze_W1:
                da=dz@self.W2.T*(a>0); self.W1-=LR*X.T@da; self.b1-=LR*da.sum(0,keepdims=True)
    def acc(self,X,y): return np.mean(self.pred(X)==y)

# Phase 1: Independent training
experts = [Expert(eid) for eid in range(5)]
for eid, e in enumerate(experts):
    e.train(datasets[eid][0], datasets[eid][1])

alone_acc = [e.acc(datasets[eid][2], datasets[eid][3]) for eid, e in enumerate(experts)]

# Phase 2: Mutual improvement using KNOWN intersections
improved_W1 = [e.W1.copy() for e in experts]

# Multiple rounds of improvement
for round_num in range(2):
    for (i, j), shared_feats in INTERSECTIONS.items():
        # Expert with MORE data teaches the other on shared features
        teacher, student = (i, j) if N_TRAIN[i] > N_TRAIN[j] else (j, i)
        alpha = 0.25  # interpolation weight
        for k in shared_feats:
            improved_W1[student][k, :] = (1-alpha)*improved_W1[student][k, :] + alpha*improved_W1[teacher][k, :]

# Phase 3: Apply + fine-tune
improved_experts = []
for eid in range(5):
    e = Expert(eid)
    e.W1 = improved_W1[eid]; e.b1 = experts[eid].b1.copy()
    e.W2 = experts[eid].W2.copy(); e.b2 = experts[eid].b2.copy()
    e.train(datasets[eid][0], datasets[eid][1], epochs=60, freeze_W1=True)
    improved_experts.append(e)

impr_acc = [e.acc(datasets[eid][2], datasets[eid][3]) for eid, e in enumerate(improved_experts)]

# Print
print("="*70)
print("PARADIGM ECOSYSTEM: Known Intersections, Mutual Improvement")
print("="*70)
print(f"\nIntersections:")
for (i,j), feats in INTERSECTIONS.items():
    print(f"  E{i} ↔ E{j}: {feats} ({len(feats)} features)")

print(f"\n{'Expert':<8} {'Data':>6} {'Features':<25} {'Alone':>8} {'Improved':>10} {'Δ':>8}")
print("-"*70)
for eid in range(5):
    feats_str = str(FEATURES[eid])[:23]
    print(f"E{eid:<7} {N_TRAIN[eid]:>6} {feats_str:<25} {alone_acc[eid]:>8.3f} {impr_acc[eid]:>10.3f} {impr_acc[eid]-alone_acc[eid]:>+8.3f}")

# Who improved whom?
print(f"\nMutual improvement contributions:")
for (i,j), feats in INTERSECTIONS.items():
    teacher, student = (i, j) if N_TRAIN[i] > N_TRAIN[j] else (j, i)
    delta = impr_acc[student] - alone_acc[student]
    print(f"  E{teacher}({N_TRAIN[teacher]}s) → E{student}({N_TRAIN[student]}s): {len(feats)} shared features, Δ={delta:+.3f}")

avg_a = np.mean(alone_acc); avg_i = np.mean(impr_acc)
print(f"\nAverage: {avg_a:.3f} → {avg_i:.3f}  (Δ={avg_i-avg_a:+.3f})")
