#!/usr/bin/env python3
"""Split-MNIST 5-way — FAIR comparison (equal compute budget)"""
import numpy as np

data = np.load('/tmp/mnist.npz')
X_tr_all, y_tr_all = data['X_tr'], data['y_tr']
X_te_all, y_te_all = data['X_te'], data['y_te']

N_TASKS, CLASSES_PER = 2, 5
N_TRAIN, N_TEST = 1500, 800
N_PIXELS, HIDDEN, LR = 784, 64, 0.01

X_train, y_train, X_test, y_test = [], [], [], []
for t in range(N_TASKS):
    s = t * CLASSES_PER; e = s + CLASSES_PER
    mt = (y_tr_all >= s) & (y_tr_all < e)
    X_train.append(X_tr_all[mt][:N_TRAIN])
    y_train.append(y_tr_all[mt][:N_TRAIN] - s)
    mt2 = (y_te_all >= s) & (y_te_all < e)
    X_test.append(X_te_all[mt2][:N_TEST])
    y_test.append(y_te_all[mt2][:N_TEST] - s)

N_OUT = N_TASKS * CLASSES_PER  # 10 total outputs

class MLP:
    def __init__(self, seed=42):
        rng = np.random.RandomState(seed)
        s = 0.05
        self.W1 = rng.randn(N_PIXELS, HIDDEN)*s; self.b1 = np.zeros((1,HIDDEN))
        self.W2 = rng.randn(HIDDEN, N_OUT)*s; self.b2 = np.zeros((1,N_OUT))
    def fwd(self, X):
        a = np.maximum(0, X@self.W1+self.b1)
        z = a@self.W2+self.b2
        e = np.exp(z-z.max(1,keepdims=True))
        return e/e.sum(1,keepdims=True)
    def acc(self, X, y, tid):
        return np.mean(self.fwd(X)[:,tid*CLASSES_PER:(tid+1)*CLASSES_PER].argmax(1)==y)

class BinMLP:
    def __init__(self, seed=42):
        rng = np.random.RandomState(seed)
        s = 0.05
        self.W1 = rng.randn(N_PIXELS, HIDDEN)*s; self.b1 = np.zeros((1,HIDDEN))
        self.W2 = rng.randn(HIDDEN, CLASSES_PER)*s; self.b2 = np.zeros((1,CLASSES_PER))
    def fwd(self, X):
        a = np.maximum(0, X@self.W1+self.b1)
        z = a@self.W2+self.b2
        e = np.exp(z-z.max(1,keepdims=True))
        return e/e.sum(1,keepdims=True)
    def acc(self, X, y):
        return np.mean(self.fwd(X).argmax(1)==y)

# ─── Equal compute budget ───
# SGD: 2 tasks × N_epochs epochs × N_TRAIN samples
# PS: 2 experts × N_epochs epochs × N_TRAIN samples (SAME total)
# But PS experts train on HALF the data (only their task)
# So PS gets N_TASKS × LESS data per expert, but each expert is smaller (N_OUT/2 outputs)
# FAIR: same epochs, same LR

N_EPOCHS = 160  # per task for SGD, per expert for PS

def run_sgd(seed, epochs=N_EPOCHS):
    m = MLP(seed); res = []
    for t in range(N_TASKS):
        Xb, yb = X_train[t], y_train[t]
        yo = np.eye(N_OUT)[yb + t*CLASSES_PER]
        for _ in range(epochs):
            p = m.fwd(Xb); N = len(Xb)
            a = np.maximum(0, Xb@m.W1+m.b1)
            dz = (p-yo)/N
            m.W2 -= LR*a.T@dz; m.b2 -= LR*dz.sum(0,keepdims=True)
            da = dz@m.W2.T*(a>0)
            m.W1 -= LR*Xb.T@da; m.b1 -= LR*da.sum(0,keepdims=True)
        res.append([m.acc(X_test[i],y_test[i],i) for i in range(t+1)])
    return res

def run_lwf(seed, epochs=N_EPOCHS):
    m = MLP(seed); res = []
    for t in range(N_TASKS):
        Xb, yb = X_train[t], y_train[t]
        old_logits = None
        if t > 0:
            old_logits = np.zeros((len(Xb), t*CLASSES_PER))
            for pt in range(t):
                old_logits[:,pt*CLASSES_PER:(pt+1)*CLASSES_PER] = m.fwd(Xb)[:,pt*CLASSES_PER:(pt+1)*CLASSES_PER]
        yo = np.eye(N_OUT)[yb + t*CLASSES_PER]
        for _ in range(epochs):
            p = m.fwd(Xb); N = len(Xb)
            a = np.maximum(0, Xb@m.W1+m.b1)
            dz = (p-yo)/N
            if t > 0 and old_logits is not None:
                co = p[:,:t*CLASSES_PER]
                os = np.exp(old_logits/2.0); os = os/(os.sum(1,keepdims=True)+1e-8)
                cs = np.exp(co/2.0); cs = cs/(cs.sum(1,keepdims=True)+1e-8)
                dz[:,:t*CLASSES_PER] += (cs-os)/N
            m.W2 -= LR*a.T@dz; m.b2 -= LR*dz.sum(0,keepdims=True)
            da = dz@m.W2.T*(a>0)
            m.W1 -= LR*Xb.T@da; m.b1 -= LR*da.sum(0,keepdims=True)
        res.append([m.acc(X_test[i],y_test[i],i) for i in range(t+1)])
    return res

def run_ps(seed, epochs=N_EPOCHS):
    """FAIR: each expert gets epochs (same as SGD per task)."""
    experts = []; res = []
    for t in range(N_TASKS):
        e = BinMLP(seed+t)
        Xb, yb = X_train[t], y_train[t]
        for _ in range(epochs):
            p = e.fwd(Xb); N = len(Xb)
            a = np.maximum(0, Xb@e.W1+e.b1)
            yo = np.eye(CLASSES_PER)[yb]
            dz = (p-yo)/N
            e.W2 -= LR*a.T@dz; e.b2 -= LR*dz.sum(0,keepdims=True)
            da = dz@e.W2.T*(a>0)
            e.W1 -= LR*Xb.T@da; e.b1 -= LR*da.sum(0,keepdims=True)
        experts.append(e)
        res.append([experts[i].acc(X_test[i],y_test[i]) for i in range(t+1)])
    return res

# ─── Sweep epochs to show convergence ───
print("="*65)
print("SPLIT-MNIST 5-WAY — Fair Comparison (equal epochs)")
print(f"{N_TASKS} tasks × {CLASSES_PER} classes")
print("="*65)

for epochs in [20, 40, 80, 160]:
    print(f"\n--- Epochs per task: {epochs} ---")
    for name, fn, seed in [('SGD', run_sgd, 42), ('LwF', run_lwf, 42), ('PS', run_ps, 42)]:
        r = fn(seed, epochs)
        f = r[-1]
        avg, mn = np.mean(f), min(f)
        forget = r[0][0] - f[0] if len(f) > 1 else 0
        print(f"  {name:<6} avg={avg:.3f}  min={mn:.3f}  forget={forget:+.3f}  [{f[0]:.3f}, {f[1]:.3f}]")

# ─── Multiple seeds for final table ───
print(f"\n{'='*65}")
print(f"FINAL (epochs={N_EPOCHS}, 5 seeds)")
print(f"{'='*65}")
print(f"{'Method':<10} {'Avg':>7} {'±Std':>7} {'Min':>7} {'Forget':>7}")

for name, fn in [('SGD', run_sgd), ('LwF', run_lwf), ('PS', run_ps)]:
    avgs, mins, forgets = [], [], []
    for seed in [42, 99, 123, 456, 789]:
        r = fn(seed, N_EPOCHS)
        f = r[-1]
        avgs.append(np.mean(f))
        mins.append(min(f))
        forgets.append(r[0][0] - f[0] if len(f) > 1 else 0)
    print(f"{name:<10} {np.mean(avgs):>7.3f} {np.std(avgs):>7.3f} {np.mean(mins):>7.3f} {np.mean(forgets):>+7.3f}")
