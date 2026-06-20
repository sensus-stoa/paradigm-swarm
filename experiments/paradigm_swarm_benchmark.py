#!/usr/bin/env python3
"""Paradigm Swarm — Final Benchmark. Clean rewrite."""
import numpy as np

np.random.seed(42)

N_TASKS, N_TRAIN, N_TEST = 10, 150, 100
SEEDS = [42, 99, 123]
HIDDEN, EPOCHS, LR = 32, 400, 0.05

# Generate clusters in a ring
centers = [np.array([6*np.cos(2*np.pi*i/N_TASKS), 6*np.sin(2*np.pi*i/N_TASKS)]) for i in range(N_TASKS)]
X_train = [np.random.randn(N_TRAIN,2)*1.2 + c for c in centers]
X_test = [np.random.randn(N_TEST,2)*1.2 + c for c in centers]

# ═══════════════════════════════════════════════════
# SHARED MLP (used by SGD, EWC, Replay, Oracle, PackNet)
# ═══════════════════════════════════════════════════
class MLP:
    def __init__(self, seed):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(2, HIDDEN)*0.3; self.b1 = np.zeros((1,HIDDEN))
        self.W2 = rng.randn(HIDDEN, N_TASKS)*0.3; self.b2 = np.zeros((1,N_TASKS))
        self.fisher_W1 = np.zeros_like(self.W1)
        self.fisher_W2 = np.zeros_like(self.W2)
        self.mask_W2 = np.ones_like(self.W2)

    def forward(self, X):
        a1 = np.tanh(X @ self.W1 + self.b1)
        z = a1 @ self.W2 + self.b2
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def predict(self, X):
        return self.forward(X).argmax(axis=1)

    def train_step(self, X, y_oh, ewc_W1=None, ewc_W2=None, ewc_lam=0):
        p = self.forward(X); N = X.shape[0]
        a1 = np.tanh(X @ self.W1 + self.b1)
        dz2 = (p - y_oh) / N
        dW2 = a1.T @ dz2; db2 = dz2.sum(0, keepdims=True)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (1 - a1**2)
        dW1 = X.T @ dz1; db1 = dz1.sum(0, keepdims=True)
        if ewc_W1 is not None:
            dW1 += ewc_lam * self.fisher_W1 * (self.W1 - ewc_W1)
            dW2 += ewc_lam * self.fisher_W2 * (self.W2 - ewc_W2)
        dW2 *= self.mask_W2
        self.W2 -= LR * dW2; self.b2 -= LR * db2
        self.W1 -= LR * dW1; self.b1 -= LR * db1

    def update_fisher(self, X, y_oh):
        p = self.forward(X); N = X.shape[0]
        a1 = np.tanh(X @ self.W1 + self.b1)
        dz2 = (p - y_oh) / N
        dW2 = a1.T @ dz2
        da1 = dz2 @ self.W2.T
        dW1 = X.T @ (da1 * (1 - a1**2))
        self.fisher_W1 += dW1**2
        self.fisher_W2 += dW2**2

def run_oracle(seed):
    model = MLP(seed)
    Xa = np.vstack(X_train)
    ya = np.hstack([np.full(N_TRAIN,i) for i in range(N_TASKS)])
    y_oh = np.eye(N_TASKS)[ya]
    for _ in range(EPOCHS*3):
        model.train_step(Xa, y_oh)
    return [np.mean(model.predict(X_test[i]) == i) for i in range(N_TASKS)]

def run_sgd(seed):
    model = MLP(seed)
    results = []
    for t in range(N_TASKS):
        y_oh = np.eye(N_TASKS)[np.full(N_TRAIN, t)]
        for _ in range(EPOCHS):
            model.train_step(X_train[t], y_oh)
        results.append([np.mean(model.predict(X_test[i]) == i) for i in range(t+1)])
    return results

def run_ewc(seed, lam):
    model = MLP(seed)
    results = []
    for t in range(N_TASKS):
        old_W1, old_W2 = model.W1.copy(), model.W2.copy()
        y_oh = np.eye(N_TASKS)[np.full(N_TRAIN, t)]
        for _ in range(EPOCHS):
            ewc_W1 = old_W1 if t > 0 else None
            ewc_W2 = old_W2 if t > 0 else None
            model.train_step(X_train[t], y_oh, ewc_W1, ewc_W2, lam if t > 0 else 0)
        model.update_fisher(X_train[t], y_oh)
        results.append([np.mean(model.predict(X_test[i]) == i) for i in range(t+1)])
    return results

def run_replay(seed, frac=0.1):
    model = MLP(seed)
    results = []
    buf_X, buf_y = [], []
    for t in range(N_TASKS):
        n_store = int(N_TRAIN * frac)
        idx = np.random.choice(N_TRAIN, n_store, replace=False)
        buf_X.append(X_train[t][idx])
        buf_y.append(np.full(n_store, t))
        Xc = np.vstack([X_train[t]] + buf_X)
        yc = np.hstack([np.full(N_TRAIN, t)] + buf_y)
        y_oh = np.eye(N_TASKS)[yc]
        for _ in range(EPOCHS):
            model.train_step(Xc, y_oh)
        results.append([np.mean(model.predict(X_test[i]) == i) for i in range(t+1)])
    return results

def run_packnet(seed, prune=0.5):
    model = MLP(seed)
    results = []
    for t in range(N_TASKS):
        if t > 0:
            trainable = (model.mask_W2 == 1.0)
            if trainable.sum() > 5:
                n = int(trainable.sum() * prune)
                w = np.abs(model.W2) * trainable
                flat = w.flatten()
                thr = np.sort(flat[flat>0])[min(n, (flat>0).sum()-1)]
                model.mask_W2 *= (w >= thr)
        y_oh = np.eye(N_TASKS)[np.full(N_TRAIN, t)]
        for _ in range(EPOCHS):
            model.train_step(X_train[t], y_oh)
        results.append([np.mean(model.predict(X_test[i]) == i) for i in range(t+1)])
    return results

# ═══════════════════════════════════════════════════
# PARADIGM SWARM
# ═══════════════════════════════════════════════════
class Gauss:
    def __init__(self, X):
        self.m = X.mean(0); self.ic = np.linalg.inv(np.cov(X.T)+np.eye(2)*0.01)
        self.d = np.linalg.det(np.cov(X.T)+np.eye(2)*0.01)
    def dens(self, X):
        d = X - self.m
        return np.exp(-0.5*np.sum(d@self.ic*d,1))/np.sqrt((2*np.pi)**2*self.d)

def run_ps(seed):
    experts = []; results = []
    for t in range(N_TASKS):
        experts.append(Gauss(X_train[t]))
        results.append([np.mean(np.column_stack([e.dens(X_test[i]) for e in experts]).argmax(1) == i) for i in range(t+1)])
    return results

def run_ps_wrong(seed):
    experts = []; results = []
    for t in range(N_TASKS):
        experts.append(Gauss(X_train[t]))
        results.append([np.mean((np.column_stack([e.dens(X_test[i]) for e in experts]).argmax(1)+1) % len(experts) == i) for i in range(t+1)])
    return results

# ═══════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════
methods = {
    'Oracle (joint)': run_oracle,
    'SGD': run_sgd,
    'EWC λ=1000': lambda s: run_ewc(s, 1000),
    'EWC λ=5000': lambda s: run_ewc(s, 5000),
    'Replay 10%': run_replay,
    'PackNet 50%': run_packnet,
    'Paradigm Swarm': run_ps,
    'PS (wrong router)': run_ps_wrong,
}

print("="*80)
print(f"PARADIGM SWARM BENCHMARK — {N_TASKS} tasks, {len(SEEDS)} seeds")
print("="*80)

all_data = {}

for name, fn in methods.items():
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    seed_results = []
    for seed in SEEDS:
        res = fn(seed)
        seed_results.append(res)

    # For oracle: res is flat list, wrap as single step
    if name == 'Oracle (joint)':
        final = seed_results[0]  # list of 10 accuracies
        print(f"  All tasks: {' '.join(f'{a:.3f}' for a in final)}")
        print(f"  Avg: {np.mean(final):.3f}")
    else:
        # Print first seed progress
        r0 = seed_results[0]
        for step in [0, N_TASKS//2, N_TASKS-1]:
            accs = r0[step]
            stars = ''.join('*' if a < 0.3 else '' for a in accs[:step+1])
            print(f"  Step {step:2d}: {' '.join(f'{a:.3f}' for a in accs)}  avg={np.mean(accs):.3f} {stars}")

    # Compute final avg across seeds
    if name == 'Oracle (joint)':
        finals = [np.mean(r) for r in seed_results]
    else:
        finals = [np.mean(r[-1]) for r in seed_results]
    all_data[name] = {'avg': np.mean(finals), 'std': np.std(finals), 'min': np.mean([min(r[-1]) for r in seed_results]) if name != 'Oracle (joint)' else np.min(final)}
    print(f"  Avg final: {all_data[name]['avg']:.3f} ± {all_data[name]['std']:.3f}")

# ═══════════════════════════════════════════════════
# FINAL TABLE
# ═══════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"FINAL RESULTS")
print(f"{'='*80}")
print(f"{'Method':<25} {'Avg':>8} {'±Std':>8} {'Min':>8}")
print("-"*55)
best = max(d['avg'] for d in all_data.values())
for name, d in all_data.items():
    flag = "←" if d['avg'] == best else ""
    print(f"{name:<25} {d['avg']:>8.3f} {d['std']:>8.3f} {d['min']:>8.3f} {flag}")
