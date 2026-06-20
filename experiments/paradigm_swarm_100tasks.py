#!/usr/bin/env python3
"""100 tasks, single 2-output head — MASSIVE catastrophic forgetting"""
import numpy as np

np.random.seed(42)
DIM, HIDDEN, LR = 100, 64, 0.01
N_TASKS = 100
EPOCHS = 100  # per task

# Each task: 4 random informative features
# 50% positive sign, 50% negative (adversarial!)
task_features = []
for t in range(N_TASKS):
    feats = sorted(np.random.choice(DIM, 4, replace=False))
    sign = 1 if t % 2 == 0 else -1  # alternate signs → interference!
    task_features.append((feats, sign))

N_SAMPLES = 500
X_base = np.random.randn(N_SAMPLES, DIM) * 0.3

X_train, y_train, X_test, y_test = [], [], [], []
for t in range(N_TASKS):
    X = X_base.copy(); feats, sign = task_features[t]
    X[:, feats] *= 5
    y = ((X[:, feats].sum(1)*sign) > 0).astype(int)
    X += np.random.randn(*X.shape)*0.3
    X_train.append(X[:200]); y_train.append(y[:200])
    X_test.append(X[300:350]); y_test.append(y[300:350])

class TrueSGD:
    def __init__(self):
        rng=np.random.RandomState(42); s=0.05
        self.W1=rng.randn(DIM,HIDDEN)*s; self.b1=np.zeros((1,HIDDEN))
        self.W2=rng.randn(HIDDEN,2)*s; self.b2=np.zeros((1,2))
    def fwd(self,X):
        a=np.maximum(0,X@self.W1+self.b1); z=a@self.W2+self.b2
        e=np.exp(z-z.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)
    def pred(self,X): return self.fwd(X).argmax(1)
    def train_task(self,X,y):
        yo=np.eye(2)[y]
        for _ in range(EPOCHS):
            idx=np.random.choice(len(X),min(64,len(X)),replace=False)
            Xb,yb=X[idx],y[idx]
            p=self.fwd(Xb); N=len(Xb); a=np.maximum(0,Xb@self.W1+self.b1)
            dz=(p-yo[idx])/N; self.W2-=LR*a.T@dz; self.b2-=LR*dz.sum(0,keepdims=True)
            da=dz@self.W2.T*(a>0); self.W1-=LR*Xb.T@da; self.b1-=LR*da.sum(0,keepdims=True)

class Expert:
    def __init__(self, seed):
        rng=np.random.RandomState(seed); s=0.05
        self.W1=rng.randn(DIM,HIDDEN)*s; self.b1=np.zeros((1,HIDDEN))
        self.W2=rng.randn(HIDDEN,2)*s; self.b2=np.zeros((1,2))
    def fwd(self,X):
        a=np.maximum(0,X@self.W1+self.b1); z=a@self.W2+self.b2
        e=np.exp(z-z.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)
    def pred(self,X): return self.fwd(X).argmax(1)
    def train(self,X,y):
        for _ in range(EPOCHS):
            idx=np.random.choice(len(X),min(64,len(X)),replace=False)
            Xb,yb=X[idx],y[idx]
            p=self.fwd(Xb); N=len(Xb); a=np.maximum(0,Xb@self.W1+self.b1)
            yo=np.eye(2)[yb]; dz=(p-yo)/N
            self.W2-=LR*a.T@dz; self.b2-=LR*dz.sum(0,keepdims=True)
            da=dz@self.W2.T*(a>0)
            self.W1-=LR*Xb.T@da; self.b1-=LR*da.sum(0,keepdims=True)

print("="*70)
print(f"100 TASKS — Single 2-output head. {EPOCHS} epochs/task.")
print("="*70)

# Swarm: train all experts
swarm = [Expert(42+t) for t in range(N_TASKS)]
for t in range(N_TASKS):
    swarm[t].train(X_train[t], y_train[t])

# TrueSGD: sequential training, measure after each task
sgd = TrueSGD()
sgd_snapshots = []  # every 10 tasks, measure ALL previous tasks

for t in range(N_TASKS):
    sgd.train_task(X_train[t], y_train[t])
    if t in [0, 9, 24, 49, 74, 99]:
        accs = [np.mean(sgd.pred(X_test[i])==y_test[i]) for i in range(t+1)]
        sgd_snapshots.append((t, accs))

# Results
print(f"\nSGD accuracy snapshots (tested on ALL previous tasks):")
for t, accs in sgd_snapshots:
    avg = np.mean(accs); mn = min(accs)
    first5 = ' '.join(f'{a:.2f}' for a in accs[:5])
    print(f"  After T{t:<3}: avg={avg:.3f} min={mn:.3f} first5=[{first5}...]")

# Final comparison
sgd_final = [np.mean(sgd.pred(X_test[i])==y_test[i]) for i in range(N_TASKS)]
swarm_final = [np.mean(swarm[i].pred(X_test[i])==y_test[i]) for i in range(N_TASKS)]

print(f"\n{'='*70}")
print(f"FINAL (100 tasks):")
print(f"  SGD avg:   {np.mean(sgd_final):.3f}")
print(f"  Swarm avg: {np.mean(swarm_final):.3f}")
print(f"  GAP:       {np.mean(swarm_final)-np.mean(sgd_final):+.3f}")
print(f"  SGD min:   {min(sgd_final):.3f}")
print(f"  Swarm min: {min(swarm_final):.3f}")

# How many tasks did SGD destroy?
sgd_dead = sum(1 for a in sgd_final if a < 0.55)
swarm_dead = sum(1 for a in swarm_final if a < 0.55)
print(f"\n  Tasks < 0.55 (near random): SGD={sgd_dead}  Swarm={swarm_dead}")

# Track first 5 tasks
print(f"\n  First 5 tasks accuracy:")
for i in range(5):
    print(f"    T{i}: SGD={sgd_final[i]:.3f}  Swarm={swarm_final[i]:.3f}")

# ─── Multi-seed summary (§4.15) ───
print(f"\n{'='*70}")
print("MULTI-SEED SUMMARY (5 seeds)")
print(f"{'='*70}")
print(f"{'Seed':<8} {'SGD avg':>8} {'Swarm avg':>10} {'Gap':>8}")
print("-"*38)
sgd_avgs, swarm_avgs = [], []
for seed in [42, 99, 123, 456, 789]:
    rng = np.random.RandomState(seed)
    Xb = rng.randn(N_SAMPLES, DIM) * 0.3
    tfeats = []
    for t in range(N_TASKS):
        feats = sorted(rng.choice(DIM, 4, replace=False))
        sign = 1 if t % 2 == 0 else -1
        tfeats.append((feats, sign))
    Xtr_s, ytr_s, Xte_s, yte_s = [], [], [], []
    for t in range(N_TASKS):
        Xs = Xb.copy(); feats, sign = tfeats[t]
        Xs[:, feats] *= 5; ys = ((Xs[:, feats].sum(1)*sign)>0).astype(int)
        Xs += rng.randn(*Xs.shape)*0.3
        Xtr_s.append(Xs[:200]); ytr_s.append(ys[:200])
        Xte_s.append(Xs[300:350]); yte_s.append(ys[300:350])
    sw_s = [Expert(seed+t) for t in range(N_TASKS)]
    for t in range(N_TASKS): sw_s[t].train(Xtr_s[t], ytr_s[t])
    sgd_s = TrueSGD()
    for t in range(N_TASKS): sgd_s.train_task(Xtr_s[t], ytr_s[t])
    sgd_f = [np.mean(sgd_s.pred(Xte_s[i])==yte_s[i]) for i in range(N_TASKS)]
    sw_f = [np.mean(sw_s[i].pred(Xte_s[i])==yte_s[i]) for i in range(N_TASKS)]
    sgd_avgs.append(np.mean(sgd_f)); swarm_avgs.append(np.mean(sw_f))
    print(f"{seed:<8} {np.mean(sgd_f):>8.3f} {np.mean(sw_f):>10.3f} {np.mean(sw_f)-np.mean(sgd_f):>+8.3f}")
print("-"*38)
print(f"{'Mean':<8} {np.mean(sgd_avgs):>8.3f} {np.mean(swarm_avgs):>10.3f} {np.mean(swarm_avgs)-np.mean(sgd_avgs):>+8.3f}")
print(f"{'±Std':<8} {np.std(sgd_avgs):>8.3f} {np.std(swarm_avgs):>10.3f}")
