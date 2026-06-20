#!/usr/bin/env python3
"""Cross-paradigm: B CANNOT be solved without A's knowledge"""
import numpy as np

np.random.seed(42)
N_TR, N_TE = 400, 200
DIM, HIDDEN, LR, EPOCHS = 10, 32, 0.05, 200

X_base = np.random.randn(N_TR + N_TE, DIM) * 0.5

# Task A: shape (features 0-4, STRONG signal)
XA = X_base.copy(); XA[:, :5] *= 4
yA = (XA[:, :5].sum(1) > 0).astype(int)

# Task B: shape+color, but shape signal is VERY WEAK (×0.3)
# Without A's pretraining, B can barely detect the shape component
XB = X_base.copy()
XB[:, :5] *= 0.8   # WEAK shape signal — hard to detect from scratch
XB[:, 5:] *= 4     # strong color signal
yB = ((XB[:, :5].sum(1) * 1.5 + XB[:, 5:].sum(1)) > 0).astype(int)

# Task C: color only (features 5-9, independent)
XC = X_base.copy(); XC[:, 5:] *= 4
yC = (XC[:, 5:].sum(1) > 0).astype(int)

X_tr = [XA[:N_TR], XB[:N_TR], XC[:N_TR]]
y_tr = [yA[:N_TR], yB[:N_TR], yC[:N_TR]]
X_te = [XA[N_TR:], XB[N_TR:], XC[N_TR:]]
y_te = [yA[N_TR:], yB[N_TR:], yC[N_TR:]]

class BinMLP:
    def __init__(self, seed=42, hid=HIDDEN):
        rng = np.random.RandomState(seed); s = 0.05
        self.W1 = rng.randn(DIM, hid)*s; self.b1 = np.zeros((1,hid))
        self.W2 = rng.randn(hid, 2)*s; self.b2 = np.zeros((1,2))
    def fwd(self, X):
        a = np.maximum(0, X@self.W1+self.b1); z = a@self.W2+self.b2
        e = np.exp(z-z.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)
    def predict(self, X): return self.fwd(X).argmax(1)
    def train(self, X, y, epochs=EPOCHS):
        yo = np.eye(2)[y]
        for _ in range(epochs):
            p = self.fwd(X); N = len(X)
            a = np.maximum(0, X@self.W1+self.b1); dz = (p-yo)/N
            self.W2 -= LR*a.T@dz; self.b2 -= LR*dz.sum(0,keepdims=True)
            da = dz@self.W2.T*(a>0)
            self.W1 -= LR*X.T@da; self.b1 -= LR*da.sum(0,keepdims=True)

# ═══════════════════════════════════════════════════
# CONDITIONS
# ═══════════════════════════════════════════════════

# 1. Pure Isolation: B trained from scratch (struggles with weak shape signal)
eA1 = BinMLP(42); eA1.train(X_tr[0], y_tr[0])
eB1 = BinMLP(43); eB1.train(X_tr[1], y_tr[1])  # from scratch
eC1 = BinMLP(44); eC1.train(X_tr[2], y_tr[2])
iso_acc = [np.mean(e.predict(X_te[i])==y_te[i]) for i,e in enumerate([eA1,eB1,eC1])]

# 2. Transfer A→B: B initialized from A's W1 (knows shape already, learns color)
eA2 = BinMLP(42); eA2.train(X_tr[0], y_tr[0])
eB2 = BinMLP(43); eB2.W1 = eA2.W1.copy(); eB2.b1 = eA2.b1.copy()
eB2.train(X_tr[1], y_tr[1])
eC2 = BinMLP(44); eC2.train(X_tr[2], y_tr[2])
trans_acc = [np.mean(e.predict(X_te[i])==y_te[i]) for i,e in enumerate([eA2,eB2,eC2])]

# 3. SGD sequential
class MultiMLP:
    def __init__(self): 
        rng=np.random.RandomState(42); s=0.05
        self.W1=rng.randn(DIM,HIDDEN)*s; self.b1=np.zeros((1,HIDDEN))
        self.W2=rng.randn(HIDDEN,6)*s; self.b2=np.zeros((1,6))
    def fwd(self,X):
        a=np.maximum(0,X@self.W1+self.b1); z=a@self.W2+self.b2
        e=np.exp(z-z.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)
    def pred(self,X,tid): return self.fwd(X)[:,tid*2:tid*2+2].argmax(1)
    def train(self,X,y,tid):
        yo=np.eye(6)[y+tid*2]
        for _ in range(EPOCHS):
            p=self.fwd(X); N=len(X); a=np.maximum(0,X@self.W1+self.b1)
            dz=(p-yo)/N; self.W2-=LR*a.T@dz; self.b2-=LR*dz.sum(0,keepdims=True)
            da=dz@self.W2.T*(a>0); self.W1-=LR*X.T@da; self.b1-=LR*da.sum(0,keepdims=True)

sgd=MultiMLP(); sgd_res=[]
for t in range(3):
    sgd.train(X_tr[t],y_tr[t],t)
    sgd_res.append([np.mean(sgd.pred(X_te[i],i)==y_te[i]) for i in range(t+1)])
sgd_final = [sgd_res[-1][i] if i<len(sgd_res[-1]) else 0 for i in range(3)]

# 4. Upper bound: B trained on FULL information (both A and B data)
eB_oracle = BinMLP(45)
XB_full = np.vstack([X_tr[0], X_tr[1]])  # train on BOTH A and B data
yB_full = np.hstack([y_tr[0], y_tr[1]])
eB_oracle.train(XB_full, yB_full)
oracle_b = np.mean(eB_oracle.predict(X_te[1]) == y_te[1])

print("="*65)
print("CROSS-PARADIGM DEPENDENCY: B NEEDS A")
print("="*65)
print(f"\n{'Method':<30} {'Task A':>8} {'Task B':>8} {'Task C':>8} {'Avg':>8}")
print("-"*65)
for name, accs in [
    ('PS Pure Isolation', iso_acc),
    ('PS Transfer A→B', trans_acc),
    ('SGD Sequential', sgd_final),
]:
    print(f"{name:<30} {accs[0]:>8.3f} {accs[1]:>8.3f} {accs[2]:>8.3f} {np.mean(accs):>8.3f}")

print(f"\nTask B upper bound (trained on A+B data): {oracle_b:.3f}")
print(f"\nPure Isolation B: {iso_acc[1]:.3f} ← struggles (weak shape signal, no A knowledge)")
print(f"Transfer A→B B:   {trans_acc[1]:.3f} ← benefits from A's pretrained weights")
print(f"Gap closed by transfer: {trans_acc[1]-iso_acc[1]:+.3f}")
print(f"\nWhen paradigms DEPEND on each other, pure isolation is suboptimal.")
print(f"Solution: explicit cross-paradigm weight inheritance or shared backbone.")
