#!/usr/bin/env python3
"""Query-specific trust: per-feature trust between experts"""
import numpy as np

np.random.seed(42)
DIM, HIDDEN, LR = 40, 64, 0.01

FEATURES = {
    0: (list(range(0,8)), 1),
    2: (list(range(10,18)), 1),
    3: (list(range(0,4))+list(range(10,14)), 1),
}
N_SAMPLES = 800
X_base = np.random.randn(N_SAMPLES, DIM) * 0.3
data = {}
for tid in [0, 2, 3]:
    X = X_base.copy(); feats, sign = FEATURES[tid]
    X[:, feats] *= 5; y = ((X[:, feats].sum(1)*sign)>0).astype(int)
    X += np.random.randn(*X.shape)*0.5
    data[tid] = (X[:400], y[:400], X[400:500], y[400:500])

class Expert:
    def __init__(self, seed):
        rng=np.random.RandomState(seed); s=0.05
        self.W1=rng.randn(DIM,HIDDEN)*s; self.b1=np.zeros((1,HIDDEN))
        self.W2=rng.randn(HIDDEN,2)*s; self.b2=np.zeros((1,2))
        self.feat_imp=np.zeros(DIM)
        # Query-specific trust: trust[other_eid][feature] = (correct, total)
        self.feat_trust = {}  # other_eid -> {feature: [correct, total]}
    def fwd(self,X):
        a=np.maximum(0,X@self.W1+self.b1); z=a@self.W2+self.b2
        e=np.exp(z-z.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)
    def pred(self,X): return self.fwd(X).argmax(1)
    def prob(self,X): return self.fwd(X)[0]
    def conf(self,X): return self.fwd(X).max(axis=1)[0]
    def train(self,X,y,epochs=400):
        for _ in range(epochs):
            idx=np.random.choice(len(X),min(64,len(X)),replace=False)
            Xb,yb=X[idx],y[idx]
            p=self.fwd(Xb); N=len(Xb); a=np.maximum(0,Xb@self.W1+self.b1)
            yo=np.eye(2)[yb]; dz=(p-yo)/N
            self.W2-=LR*a.T@dz; self.b2-=LR*dz.sum(0,keepdims=True)
            da=dz@self.W2.T*(a>0)
            self.feat_imp+=np.abs(Xb.T@da).sum(axis=1)*0.001
            self.W1-=LR*Xb.T@da; self.b1-=LR*da.sum(0,keepdims=True)
    def feature_trust(self, other_eid, feature):
        if other_eid not in self.feat_trust: return 0.5
        if feature not in self.feat_trust[other_eid]: return 0.5
        c, t = self.feat_trust[other_eid][feature]
        return c / max(1, t)
    def update_feature_trust(self, other_eid, feature, agreed):
        if other_eid not in self.feat_trust:
            self.feat_trust[other_eid] = {}
        if feature not in self.feat_trust[other_eid]:
            self.feat_trust[other_eid][feature] = [0, 0]
        self.feat_trust[other_eid][feature][0] += 1 if agreed else 0
        self.feat_trust[other_eid][feature][1] += 1

# E0, E2: fully trained. E3: underfit (15 examples)
E0 = Expert(42); E0.train(data[0][0], data[0][1], epochs=1000)
E2 = Expert(44); E2.train(data[2][0], data[2][1], epochs=1000)
E3 = Expert(46); E3.train(data[3][0][:20], data[3][1][:20], epochs=500)  # 20 training examples
for e in [E0, E2, E3]:
    e.feat_imp = e.feat_imp / (e.feat_imp.max() + 1e-8)

# QUERY-SPECIFIC trust calibration — use TRUE labels, not E3's predictions
X3_calib = data[3][0][50:150]; y3_calib = data[3][1][50:150]  # 100 calibration examples
for i in range(len(X3_calib)):
    x = X3_calib[i:i+1]; xf = x[0]
    true_label = y3_calib[i]  # GROUND TRUTH
    active_f = np.where(np.abs(xf) > 1.0)[0]
    for other_eid, other_e in [(0, E0), (2, E2)]:
        other_pred = other_e.pred(x)[0]
        agreed = (other_pred == true_label)  # compare to TRUTH
        for f in active_f:
            E3.update_feature_trust(other_eid, f, agreed)

# Directory
directory = {}
for f in range(DIM):
    owners = [(eid, e.feat_imp[f]) for eid, e in [(0,E0),(2,E2),(3,E3)] if e.feat_imp[f] > 0.3]
    if owners: directory[f] = sorted(owners, key=lambda x: -x[1])

X3_test, y3_test = data[3][2], data[3][3]
CONF_THR = 0.60
TRUST_THR = 0.55

def query_specific_defer(x):
    my_conf = E3.conf(x)
    if my_conf >= CONF_THR:
        return E3.pred(x)[0]
    
    xf = x[0]; active = np.where(np.abs(xf) > 1.0)[0]
    
    # Score each potential consultee by feature-specific trust on active features
    best_eid, best_score = -1, -1
    for other_eid in [0, 2]:
        score = 0; count = 0
        for f in active:
            ft = E3.feature_trust(other_eid, f)
            if ft >= TRUST_THR:
                imp = directory.get(f, [(0,0)])[0][1] if f in directory else 0
                score += ft * imp
                count += 1
        if count > 0:
            score /= count
            if score > best_score:
                best_score, best_eid = score, other_eid
    
    if best_eid >= 0:
        experts_map = {0: E0, 2: E2}
        return experts_map[best_eid].pred(x)[0]
    return E3.pred(x)[0]

print("="*65)
print("QUERY-SPECIFIC TRUST: Per-feature calibration")
print("="*65)

# Show feature-trust for E3→E0 and E3→E2 on key features
print(f"\nE3 feature-trust (features where calibrated):")
for other_eid, name in [(0, 'E0'), (2, 'E2')]:
    for f in [0,1,2,3,10,11,12,13]:
        ft = E3.feature_trust(other_eid, f)
        if ft != 0.5:
            print(f"  E3→{name} on feature {f}: {ft:.2f}")

alone_preds = np.array([E3.pred(X3_test[i:i+1])[0] for i in range(len(X3_test))])
defer_preds = np.array([query_specific_defer(X3_test[i:i+1]) for i in range(len(X3_test))])
confs = np.array([E3.conf(X3_test[i:i+1]) for i in range(len(X3_test))])

alone_acc = np.mean(alone_preds == y3_test)
defer_acc = np.mean(defer_preds == y3_test)
low = confs < CONF_THR; high = confs >= CONF_THR

print(f"\nOverall: Alone={alone_acc:.3f}  QS-Defer={defer_acc:.3f}  Δ={defer_acc-alone_acc:+.3f}")
print(f"Low conf (<{CONF_THR}, {low.sum()}q): Alone={np.mean(alone_preds[low]==y3_test[low]):.3f}  Defer={np.mean(defer_preds[low]==y3_test[low]):.3f}")
print(f"High conf (≥{CONF_THR}, {high.sum()}q): Alone={np.mean(alone_preds[high]==y3_test[high]):.3f}  Defer={np.mean(defer_preds[high]==y3_test[high]):.3f}")

# Count changes
changes = alone_preds != defer_preds
improved = sum(1 for i in np.where(changes)[0] if defer_preds[i]==y3_test[i])
harmed = sum(1 for i in np.where(changes)[0] if defer_preds[i]!=y3_test[i])
print(f"\nDeferral changed {changes.sum()} predictions: improved={improved}, harmed={harmed}")
