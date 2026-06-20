#!/usr/bin/env python3
"""TRUE catastrophic forgetting: single 2-output head for ALL 5 tasks"""
import numpy as np

np.random.seed(42)
DIM, HIDDEN, LR = 40, 64, 0.01
N_TASKS = 5

FEATURES = {
    0: (list(range(0,8)), 1),
    1: (list(range(0,8)), -1),
    2: (list(range(10,18)), 1),
    3: (list(range(0,4))+list(range(10,14)), 1),
    4: (list(range(20,28)), 1),
}

N_SAMPLES = 800
X_base = np.random.randn(N_SAMPLES, DIM) * 0.3
X_train, y_train, X_test, y_test = [], [], [], []
for tid in range(N_TASKS):
    X = X_base.copy(); feats, sign = FEATURES[tid]
    X[:, feats] *= 5; y = ((X[:, feats].sum(1)*sign)>0).astype(int)
    X += np.random.randn(*X.shape)*0.5
    X_train.append(X[:400]); y_train.append(y[:400])
    X_test.append(X[300:400]); y_test.append(y[300:400])

class TrueSGD:
    """SGD with SINGLE 2-output head — TRUE forgetting"""
    def __init__(self):
        rng=np.random.RandomState(42); s=0.05
        self.W1=rng.randn(DIM,HIDDEN)*s; self.b1=np.zeros((1,HIDDEN))
        self.W2=rng.randn(HIDDEN,2)*s; self.b2=np.zeros((1,2))
    def fwd(self,X):
        a=np.maximum(0,X@self.W1+self.b1); z=a@self.W2+self.b2
        e=np.exp(z-z.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)
    def pred(self,X): return self.fwd(X).argmax(1)
    def train_task(self,X,y,epochs):
        yo=np.eye(2)[y]
        for _ in range(epochs):
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
    def train(self,X,y,epochs=1000):
        for _ in range(epochs):
            idx=np.random.choice(len(X),min(64,len(X)),replace=False)
            Xb,yb=X[idx],y[idx]
            p=self.fwd(Xb); N=len(Xb); a=np.maximum(0,Xb@self.W1+self.b1)
            yo=np.eye(2)[yb]; dz=(p-yo)/N
            self.W2-=LR*a.T@dz; self.b2-=LR*dz.sum(0,keepdims=True)
            da=dz@self.W2.T*(a>0)
            self.W1-=LR*Xb.T@da; self.b1-=LR*da.sum(0,keepdims=True)

# Swarm
swarm = [Expert(42+i) for i in range(N_TASKS)]
for tid in range(N_TASKS):
    swarm[tid].train(X_train[tid], y_train[tid])

# True SGD — single head, TRUE forgetting
sgd = TrueSGD()
sgd_history = []

for tid in range(N_TASKS):
    sgd.train_task(X_train[tid], y_train[tid], epochs=1000)
    accs = [np.mean(sgd.pred(X_test[i])==y_test[i]) for i in range(tid+1)]
    sgd_history.append(accs)

print("="*75)
print("CATASTROPHIC FORGETTING: Single 2-output head for ALL 5 tasks")
print("="*75)

print(f"\nSGD ACCURACY MATRIX (single output head — TRUE forgetting):")
print(f"{'After':<10}", end="")
for i in range(N_TASKS):
    print(f"{'T'+str(i):>8}", end="")
print(f"  {'Avg':>8}  {'T0 drop':>10}")
print("-"*70)

sgd_t0_start = None
for step, accs in enumerate(sgd_history):
    row = ' '.join(f'{a:>8.3f}' for a in accs) + '  -   ' * (N_TASKS-len(accs))
    avg = np.mean(accs)
    t0 = accs[0]
    if step == 0: sgd_t0_start = t0
    t0_drop = sgd_t0_start - t0
    marker = " ← FORGOTTEN!" if t0_drop > 0.2 else ""
    print(f"T{step:<9} {row} {avg:>8.3f} {t0_drop:>+10.3f}{marker}")

# Final comparison
sgd_final = sgd_history[-1]
swarm_final = [np.mean(swarm[i].pred(X_test[i])==y_test[i]) for i in range(N_TASKS)]

print(f"\n{'='*75}")
print(f"FINAL: Swarm vs TrueSGD (single head)")
print(f"{'='*75}")
print(f"{'Task':<10} {'Swarm':>8} {'TrueSGD':>8} {'Δ':>8}")
print("-"*40)
for tid in range(N_TASKS):
    s = swarm_final[tid]; g = sgd_final[tid]
    stars = " ★" if g < 0.6 else ""
    print(f"T{tid:<9} {s:>8.3f} {g:>8.3f} {s-g:>+8.3f}{stars}")

print(f"\nSGD T0: {sgd_t0_start:.3f} → {sgd_final[0]:.3f} (FORGOT {sgd_t0_start-sgd_final[0]:.3f})")
print(f"Swarm T0: {swarm_final[0]:.3f} (unchanged)")
print(f"\nSGD Avg: {np.mean(sgd_final):.3f} | Swarm Avg: {np.mean(swarm_final):.3f} | Gap: {np.mean(swarm_final)-np.mean(sgd_final):+.3f}")
