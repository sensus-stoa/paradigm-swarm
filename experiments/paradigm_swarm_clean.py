#!/usr/bin/env python3
"""Extended run: 50-2000 epochs, clean swarm vs SGD"""
import numpy as np

np.random.seed(42)
DIM, HIDDEN, LR = 40, 64, 0.01
N_TASKS = 4
FEATURES = {
    0: (list(range(0,8)), 1),
    1: (list(range(0,8)), -1),
    2: (list(range(10,18)), 1),
    3: (list(range(0,4))+list(range(10,14)), 1),
}
N_SAMPLES = 800
X_base = np.random.randn(N_SAMPLES, DIM) * 0.3
X_train, y_train, X_test, y_test = [], [], [], []
for tid in range(N_TASKS):
    X = X_base.copy(); feats, sign = FEATURES[tid]
    X[:, feats] *= 5; y = ((X[:, feats].sum(1)*sign)>0).astype(int)
    X += np.random.randn(*X.shape)*0.3
    X_train.append(X[:400]); y_train.append(y[:400])
    X_test.append(X[400:500]); y_test.append(y[400:500])

class Expert:
    def __init__(self, seed):
        rng=np.random.RandomState(seed); s=0.05
        self.W1=rng.randn(DIM,HIDDEN)*s; self.b1=np.zeros((1,HIDDEN))
        self.W2=rng.randn(HIDDEN,2)*s; self.b2=np.zeros((1,2))
        self.feat_imp=np.zeros(DIM)
    def fwd(self,X):
        a=np.maximum(0,X@self.W1+self.b1); z=a@self.W2+self.b2
        e=np.exp(z-z.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)
    def conf(self,X): return self.fwd(X).max(axis=1)
    def pred(self,X): return self.fwd(X).argmax(1)
    def train_epoch(self,X,y,epochs=50):
        for _ in range(epochs):
            idx=np.random.choice(len(X),64,replace=False)
            Xb,yb=X[idx],y[idx]
            p=self.fwd(Xb); N=len(Xb); a=np.maximum(0,Xb@self.W1+self.b1)
            yo=np.eye(2)[yb]; dz=(p-yo)/N
            self.W2-=LR*a.T@dz; self.b2-=LR*dz.sum(0,keepdims=True)
            da=dz@self.W2.T*(a>0)
            self.feat_imp+=np.abs(Xb.T@da).sum(axis=1)*0.001
            self.W1-=LR*Xb.T@da; self.b1-=LR*da.sum(0,keepdims=True)

class SGDNet:
    def __init__(self):
        rng=np.random.RandomState(42); s=0.05
        self.W1=rng.randn(DIM,HIDDEN)*s; self.b1=np.zeros((1,HIDDEN))
        self.W2=rng.randn(HIDDEN,8)*s; self.b2=np.zeros((1,8))
    def fwd(self,X):
        a=np.maximum(0,X@self.W1+self.b1); z=a@self.W2+self.b2
        e=np.exp(z-z.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)
    def pred(self,X,tid): return self.fwd(X)[:,tid*2:tid*2+2].argmax(1)
    def train_epoch(self,X,y,tid,epochs=50):
        yo=np.eye(8)[y+tid*2]
        for _ in range(epochs):
            idx=np.random.choice(len(X),64,replace=False)
            Xb,yb=X[idx],y[idx]
            p=self.fwd(Xb); N=len(Xb); a=np.maximum(0,Xb@self.W1+self.b1)
            dz=(p-yo[idx])/N; self.W2-=LR*a.T@dz; self.b2-=LR*dz.sum(0,keepdims=True)
            da=dz@self.W2.T*(a>0); self.W1-=LR*Xb.T@da; self.b1-=LR*da.sum(0,keepdims=True)

print("epochs  SGD_avg  SGD_T0  SGD_T1  SGD_T2  SGD_T3  SWARM_avg  SWARM_T0  SWARM_T1  SWARM_T2  SWARM_T3  Winner")
for epochs in [50, 100, 200, 400, 800, 1200, 1600, 2000]:
    sgd = SGDNet()
    for tid in range(N_TASKS):
        sgd.train_epoch(X_train[tid], y_train[tid], tid, epochs=epochs)
    sa = [np.mean(sgd.pred(X_test[i],i)==y_test[i]) for i in range(N_TASKS)]
    
    experts = [Expert(42+i) for i in range(N_TASKS)]
    for tid in range(N_TASKS):
        experts[tid].train_epoch(X_train[tid], y_train[tid], epochs=epochs)
    sw = [np.mean(experts[tid].pred(X_test[tid])==y_test[tid]) for tid in range(N_TASKS)]
    
    w = "SWARM" if np.mean(sw)>np.mean(sa) else "SGD"
    print(f"{epochs:<7} {np.mean(sa):.3f}     {sa[0]:.3f}    {sa[1]:.3f}    {sa[2]:.3f}    {sa[3]:.3f}     {np.mean(sw):.3f}       {sw[0]:.3f}     {sw[1]:.3f}     {sw[2]:.3f}     {sw[3]:.3f}     {w}")

# Feature detection accuracy
print(f"\nFeature detection (all experts, all epochs):")
for epochs in [800, 2000]:
    experts = [Expert(42+i) for i in range(N_TASKS)]
    for tid in range(N_TASKS):
        experts[tid].train_epoch(X_train[tid], y_train[tid], epochs=epochs)
    correct = 0
    for tid, e in enumerate(experts):
        top5 = set(np.argsort(e.feat_imp)[-5:])
        true = set(FEATURES[tid][0])
        correct += len(top5 & true)
    print(f"  ep={epochs}: {correct}/20 correct features in top-5")

# ─── Multi-seed summary (§4.12) ───
print(f"\n{'='*70}")
print("MULTI-SEED SUMMARY (5 seeds, epoch=2000)")
print(f"{'='*70}")
print(f"{'Seed':<8} {'SGD avg':>8} {'Swarm avg':>10} {'Winner':>8}")
print("-"*40)
sgd_means, swarm_means = [], []
for s in [42, 99, 123, 456, 789]:
    np.random.seed(s)
    X_base_s = np.random.randn(N_SAMPLES, DIM) * 0.3
    X_tr_s, y_tr_s, X_te_s, y_te_s = [], [], [], []
    for tid in range(N_TASKS):
        Xs = X_base_s.copy(); feats, sign = FEATURES[tid]
        Xs[:, feats] *= 5; ys = ((Xs[:, feats].sum(1)*sign)>0).astype(int)
        Xs += np.random.randn(*Xs.shape)*0.3
        X_tr_s.append(Xs[:400]); y_tr_s.append(ys[:400])
        X_te_s.append(Xs[400:500]); y_te_s.append(ys[400:500])
    sgd_s = SGDNet()
    for tid in range(N_TASKS):
        sgd_s.train_epoch(X_tr_s[tid], y_tr_s[tid], tid, epochs=2000)
    sa_s = [np.mean(sgd_s.pred(X_te_s[i],i)==y_te_s[i]) for i in range(N_TASKS)]
    experts_s = [Expert(s+i) for i in range(N_TASKS)]
    for tid in range(N_TASKS):
        experts_s[tid].train_epoch(X_tr_s[tid], y_tr_s[tid], epochs=2000)
    sw_s = [np.mean(experts_s[tid].pred(X_te_s[tid])==y_te_s[tid]) for tid in range(N_TASKS)]
    sgd_means.append(np.mean(sa_s)); swarm_means.append(np.mean(sw_s))
    w_s = "SWARM" if np.mean(sw_s)>np.mean(sa_s) else "SGD"
    print(f"{s:<8} {np.mean(sa_s):>8.3f} {np.mean(sw_s):>10.3f} {w_s:>8}")
print("-"*40)
print(f"{'Mean':<8} {np.mean(sgd_means):>8.3f} {np.mean(swarm_means):>10.3f}")
print(f"{'±Std':<8} {np.std(sgd_means):>8.3f} {np.std(swarm_means):>10.3f}")
