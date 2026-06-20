#!/usr/bin/env python3
"""FINAL: Strong Ensemble helps weak experts. SGD destroyed by adversarial."""
import numpy as np

np.random.seed(42)
DIM, HIDDEN, LR = 60, 64, 0.01
N_A, N_B = 25, 25

X_base = np.random.randn(600, DIM) * 0.3

X_train_A, y_train_A, X_test_A, y_test_A = [], [], [], []
X_train_B, y_train_B, X_test_B, y_test_B = [], [], [], []

for t in range(N_A):
    X = X_base.copy(); X[:, :8] *= 5
    sign = 1 if t%2==0 else -1
    y = ((X[:,:8].sum(1)*sign)>0).astype(int)
    X += np.random.randn(*X.shape)*0.5
    X_train_A.append(X[:200]); y_train_A.append(y[:200])
    X_test_A.append(X[300:370]); y_test_A.append(y[300:370])

for t in range(N_B):
    X = X_base.copy(); X[:, :4] *= 5; X[:, 20:24] *= 2
    y = ((X[:,:4].sum(1)+X[:,20:24].sum(1)*0.3)>0).astype(int)
    X += np.random.randn(*X.shape)*0.5
    X_train_B.append(X[:15]); y_train_B.append(y[:15])
    X_test_B.append(X[300:370]); y_test_B.append(y[300:370])

class Expert:
    def __init__(self, seed):
        rng=np.random.RandomState(seed); s=0.05
        self.W1=rng.randn(DIM,HIDDEN)*s; self.b1=np.zeros((1,HIDDEN))
        self.W2=rng.randn(HIDDEN,2)*s; self.b2=np.zeros((1,2))
    def fwd(self,X):
        a=np.maximum(0,X@self.W1+self.b1); z=a@self.W2+self.b2
        e=np.exp(z-z.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)
    def pred(self,X): return self.fwd(X).argmax(1)
    def prob(self,X): return self.fwd(X)[0]
    def conf(self,X): return self.fwd(X).max(axis=1)[0]
    def train(self,X,y,epochs=200):
        for _ in range(epochs):
            idx=np.random.choice(len(X),min(64,len(X)),replace=False)
            Xb,yb=X[idx],y[idx]
            p=self.fwd(Xb); N=len(Xb); a=np.maximum(0,Xb@self.W1+self.b1)
            yo=np.eye(2)[yb]; dz=(p-yo)/N
            self.W2-=LR*a.T@dz; self.b2-=LR*dz.sum(0,keepdims=True)
            da=dz@self.W2.T*(a>0)
            self.W1-=LR*Xb.T@da; self.b1-=LR*da.sum(0,keepdims=True)

# Strong ensemble: one expert per Category A task
print("Training 25 Strong Experts...")
strong = [Expert(42+t) for t in range(N_A)]
for t in range(N_A):
    strong[t].train(X_train_A[t], y_train_A[t], epochs=300)

# Weak experts: one per Category B task
print("Training 25 Weak Experts...")
weak = [Expert(100+t) for t in range(N_B)]
for t in range(N_B):
    weak[t].train(X_train_B[t], y_train_B[t], epochs=50)

# Trust: each weak expert evaluates ALL strong experts on its own data
print("Calibrating trust...")
trust = {}  # weak_t -> {strong_id: accuracy}
for t in range(N_B):
    Xi, yi = X_train_B[t], y_train_B[t]
    trust[t] = {}
    for s_id in range(N_A):
        trust[t][s_id] = np.mean(strong[s_id].pred(Xi) == yi)

# SGD: single 2-output head, 50 tasks
class SGD:
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
        for _ in range(200):
            idx=np.random.choice(len(X),min(64,len(X)),replace=False)
            Xb,yb=X[idx],y[idx]
            p=self.fwd(Xb); N=len(Xb); a=np.maximum(0,Xb@self.W1+self.b1)
            dz=(p-yo[idx])/N; self.W2-=LR*a.T@dz; self.b2-=LR*dz.sum(0,keepdims=True)
            da=dz@self.W2.T*(a>0); self.W1-=LR*Xb.T@da; self.b1-=LR*da.sum(0,keepdims=True)

print("Training SGD...")
sgd = SGD()
all_X = X_train_A + X_train_B; all_y = y_train_A + y_train_B
order = list(range(50)); np.random.shuffle(order)
for idx in order: sgd.train_task(all_X[idx], all_y[idx])

# ═══════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════
CONF_THR, TRUST_THR = 0.60, 0.65

# Category A: strong ensemble
strong_acc = [np.mean(strong[t].pred(X_test_A[t])==y_test_A[t]) for t in range(N_A)]

# Category B: alone vs consult vs sgd
weak_alone = [np.mean(weak[t].pred(X_test_B[t])==y_test_B[t]) for t in range(N_B)]
weak_consult = []
corrections = 0

for t in range(N_B):
    Xt, yt = X_test_B[t], y_test_B[t]
    preds = []
    for i in range(len(Xt)):
        x = Xt[i:i+1]
        if weak[t].conf(x) >= CONF_THR:
            preds.append(weak[t].pred(x)[0])
        else:
            # Find trusted strong experts
            trusted = [(s_id, trust[t][s_id]) for s_id in trust[t] if trust[t][s_id] >= TRUST_THR]
            if trusted:
                best_s = max(trusted, key=lambda x: x[1])[0]
                pred = strong[best_s].pred(x)[0]
                if pred == yt[i] and weak[t].pred(x)[0] != yt[i]:
                    corrections += 1
                preds.append(pred)
            else:
                preds.append(weak[t].pred(x)[0])
    weak_consult.append(np.mean(np.array(preds)==yt))

sgd_A = [np.mean(sgd.pred(X_test_A[t])==y_test_A[t]) for t in range(N_A)]
sgd_B = [np.mean(sgd.pred(X_test_B[t])==y_test_B[t]) for t in range(N_B)]

print("="*65)
print("FINAL: Strong Ensemble + Weak Experts vs SGD")
print("="*65)
print(f"\nCategory A (25 tasks, strong ensemble):")
print(f"  Strong:  avg={np.mean(strong_acc):.3f}  min={min(strong_acc):.3f}")
print(f"  SGD:     avg={np.mean(sgd_A):.3f}  min={min(sgd_A):.3f}")

print(f"\nCategory B (25 tasks, weak, consults strong):")
print(f"  Alone:   avg={np.mean(weak_alone):.3f}  min={min(weak_alone):.3f}")
print(f"  Consult: avg={np.mean(weak_consult):.3f}  min={min(weak_consult):.3f}")
print(f"  SGD:     avg={np.mean(sgd_B):.3f}  min={min(sgd_B):.3f}")

print(f"\nConsult Δ: {np.mean(weak_consult)-np.mean(weak_alone):+.3f}")
print(f"Corrections: {corrections}")
print(f"Avg trusted strong experts: {np.mean([sum(1 for s in trust[t] if trust[t][s]>=TRUST_THR) for t in range(N_B)]):.1f}")
print(f"\nSGD overall:  {np.mean(sgd_A+sgd_B):.3f}")
print(f"Swarm overall: {np.mean(strong_acc+weak_consult):.3f}")
