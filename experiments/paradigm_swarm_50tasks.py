#!/usr/bin/env python3
"""50 tasks — PS with MORE epochs (fair: no interference = can train longer)"""
import numpy as np

N_TASKS = 50
N_TRAIN, N_TEST = 200, 100
DIM, HIDDEN, LR, SGD_EPOCHS = 20, 64, 0.05, 40

np.random.seed(42)
X_base = np.random.randn(N_TRAIN + N_TEST, DIM) * 0.5
task_w = [np.random.RandomState(t*100+1).randn(DIM) for t in range(N_TASKS)]
task_w = [w/np.linalg.norm(w) for w in task_w]

Xs_tr, ys_tr, Xs_te, ys_te = [], [], [], []
for t in range(N_TASKS):
    X_tr = X_base[:N_TRAIN]; y_tr = ((X_tr @ task_w[t]) > 0).astype(int)
    Xs_tr.append(X_tr.copy()); ys_tr.append(y_tr)
    X_te = X_base[N_TRAIN:]; y_te = ((X_te @ task_w[t]) > 0).astype(int)
    Xs_te.append(X_te.copy()); ys_te.append(y_te)

class TrueSGD:
    def __init__(self, seed=42):
        rng = np.random.RandomState(seed); s = 0.02
        self.W1 = rng.randn(DIM, HIDDEN)*s; self.b1 = np.zeros((1,HIDDEN))
        self.W2 = rng.randn(HIDDEN, 2)*s; self.b2 = np.zeros((1,2))
    def fwd(self, X):
        a = np.maximum(0, X@self.W1+self.b1); z = a@self.W2+self.b2
        e = np.exp(z-z.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)
    def predict(self, X): return self.fwd(X).argmax(1)
    def train(self, X, y, epochs):
        yo = np.eye(2)[y]
        for _ in range(epochs):
            p = self.fwd(X); N = len(X)
            a = np.maximum(0, X@self.W1+self.b1); dz = (p-yo)/N
            self.W2 -= LR*a.T@dz; self.b2 -= LR*dz.sum(0,keepdims=True)
            da = dz@self.W2.T*(a>0)
            self.W1 -= LR*X.T@da; self.b1 -= LR*da.sum(0,keepdims=True)

class BinMLP:
    def __init__(self, seed):
        rng = np.random.RandomState(seed); s = 0.02
        self.W1 = rng.randn(DIM, HIDDEN)*s; self.b1 = np.zeros((1,HIDDEN))
        self.W2 = rng.randn(HIDDEN, 2)*s; self.b2 = np.zeros((1,2))
    def fwd(self, X):
        a = np.maximum(0, X@self.W1+self.b1); z = a@self.W2+self.b2
        e = np.exp(z-z.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)
    def predict(self, X): return self.fwd(X).argmax(1)
    def train(self, X, y, epochs):
        yo = np.eye(2)[y]
        for _ in range(epochs):
            p = self.fwd(X); N = len(X)
            a = np.maximum(0, X@self.W1+self.b1); dz = (p-yo)/N
            self.W2 -= LR*a.T@dz; self.b2 -= LR*dz.sum(0,keepdims=True)
            da = dz@self.W2.T*(a>0)
            self.W1 -= LR*X.T@da; self.b1 -= LR*da.sum(0,keepdims=True)

def harmonic_mean(arr):
    arr = np.clip(np.array(arr), 0.01, 1.0)
    return len(arr) / np.sum(1.0/arr)

print("="*70)
print(f"50 TASKS — PS with MORE epochs (fair advantage of isolation)")
print("="*70)
print(f"{'PS epochs':<12} {'SGD avg':>8} {'PS avg':>8} {'SGD harm':>8} {'PS harm':>8} {'SGD min':>8} {'PS min':>8} {'PS win?':>8}")
print("-"*70)

for ps_epochs in [40, 80, 160, 320]:
    # SGD (always 40 epochs — can't do more or forgetting worsens)
    sgd = TrueSGD()
    for t in range(N_TASKS):
        sgd.train(Xs_tr[t], ys_tr[t], SGD_EPOCHS)
    sf = [np.mean(sgd.predict(Xs_te[i]) == ys_te[i]) for i in range(N_TASKS)]

    # PS (ps_epochs per expert)
    ps_experts = []
    for t in range(N_TASKS):
        e = BinMLP(42 + t)
        e.train(Xs_tr[t], ys_tr[t], ps_epochs)
        ps_experts.append(e)
    pf = [np.mean(ps_experts[i].predict(Xs_te[i]) == ys_te[i]) for i in range(N_TASKS)]

    ps_wins_avg = np.mean(pf) > np.mean(sf)
    ps_wins_harm = harmonic_mean(pf) > harmonic_mean(sf)
    print(f"{ps_epochs:<12} {np.mean(sf):>8.3f} {np.mean(pf):>8.3f} {harmonic_mean(sf):>8.3f} {harmonic_mean(pf):>8.3f} {min(sf):>8.3f} {min(pf):>8.3f} {'AVG+HARM' if ps_wins_avg and ps_wins_harm else ('AVG' if ps_wins_avg else ('HARM' if ps_wins_harm else '—')):>8}")

# Best PS configuration detail
print(f"\n{'='*70}")
print(f"BEST PS (epochs=320) vs SGD (epochs=40)")
print(f"{'='*70}")
sgd = TrueSGD()
for t in range(N_TASKS): sgd.train(Xs_tr[t], ys_tr[t], SGD_EPOCHS)
sf = [np.mean(sgd.predict(Xs_te[i]) == ys_te[i]) for i in range(N_TASKS)]

ps_experts = []
for t in range(N_TASKS):
    e = BinMLP(42 + t); e.train(Xs_tr[t], ys_tr[t], 320); ps_experts.append(e)
pf = [np.mean(ps_experts[i].predict(Xs_te[i]) == ys_te[i]) for i in range(N_TASKS)]

print(f"  Arithmetic:  SGD={np.mean(sf):.3f}  PS={np.mean(pf):.3f}  (PS {'+' if np.mean(pf)>np.mean(sf) else ''}{np.mean(pf)-np.mean(sf):+.3f})")
print(f"  Harmonic:    SGD={harmonic_mean(sf):.3f}  PS={harmonic_mean(pf):.3f}")
print(f"  Min:         SGD={min(sf):.3f}  PS={min(pf):.3f}")
print(f"  Tasks <0.55: SGD={sum(1 for a in sf if a<0.55)}  PS={sum(1 for a in pf if a<0.55)}")
print(f"  Survival:    SGD={sum(1 for a in sf if a>=0.55)/N_TASKS:.0%}  PS={sum(1 for a in pf if a>=0.55)/N_TASKS:.0%}")
print(f"\n  Total compute (gradient steps):")
print(f"    SGD: {N_TASKS}×{SGD_EPOCHS}×{N_TRAIN} = {N_TASKS*SGD_EPOCHS*N_TRAIN:,}")
print(f"    PS:  {N_TASKS}×320×{N_TRAIN} = {N_TASKS*320*N_TRAIN:,}")
print(f"  PS uses {320/SGD_EPOCHS:.0f}x more compute — isolation removes the interference ceiling.")

# ─── Multi-seed summary (§4.10) ───
print(f"\n{'='*70}")
print("MULTI-SEED SUMMARY (5 seeds, PS=320ep, SGD=40ep)")
print(f"{'='*70}")
print(f"{'Seed':<8} {'SGD avg':>8} {'PS avg':>10} {'Winner':>8}")
print("-"*40)
sgd_avgs, ps_avgs = [], []
for seed in [42, 99, 123, 456, 789]:
    rng_s = np.random.RandomState(seed)
    Xb_s = rng_s.randn(N_TRAIN + N_TEST, DIM) * 0.5
    tw_s = [rng_s.randn(DIM) for _ in range(N_TASKS)]
    tw_s = [w/np.linalg.norm(w) for w in tw_s]
    Xtr_s, ytr_s, Xte_s, yte_s = [], [], [], []
    for t in range(N_TASKS):
        Xtr_s.append(Xb_s[:N_TRAIN]); ytr_s.append(((Xb_s[:N_TRAIN] @ tw_s[t]) > 0).astype(int))
        Xte_s.append(Xb_s[N_TRAIN:]); yte_s.append(((Xb_s[N_TRAIN:] @ tw_s[t]) > 0).astype(int))
    sgd_s = TrueSGD(seed)
    for t in range(N_TASKS): sgd_s.train(Xtr_s[t], ytr_s[t], SGD_EPOCHS)
    sf_s = [np.mean(sgd_s.predict(Xte_s[i]) == yte_s[i]) for i in range(N_TASKS)]
    ps_exps = [BinMLP(seed + t) for t in range(N_TASKS)]
    for t in range(N_TASKS): ps_exps[t].train(Xtr_s[t], ytr_s[t], 320)
    pf_s = [np.mean(ps_exps[i].predict(Xte_s[i]) == yte_s[i]) for i in range(N_TASKS)]
    sgd_avgs.append(np.mean(sf_s)); ps_avgs.append(np.mean(pf_s))
    w_s = "PS" if np.mean(pf_s)>np.mean(sf_s) else "SGD"
    print(f"{seed:<8} {np.mean(sf_s):>8.3f} {np.mean(pf_s):>10.3f} {w_s:>8}")
print("-"*40)
print(f"{'Mean':<8} {np.mean(sgd_avgs):>8.3f} {np.mean(ps_avgs):>10.3f}")
print(f"{'±Std':<8} {np.std(sgd_avgs):>8.3f} {np.std(ps_avgs):>10.3f}")
