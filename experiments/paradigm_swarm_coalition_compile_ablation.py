#!/usr/bin/env python3
"""
Quick ablation: when does coalition pseudo-label training help?
Test SOFT non-linearities where each expert has PARTIAL signal.
"""
import numpy as np
np.random.seed(42)
DIM, HIDDEN = 60, 32

class Expert:
    def __init__(self, seed):
        rng=np.random.RandomState(seed); s=0.05
        self.W1=rng.randn(DIM,HIDDEN)*s; self.b1=np.zeros((1,HIDDEN))
        self.W2=rng.randn(HIDDEN,2)*s; self.b2=np.zeros((1,2))
    def forward(self,X):
        a=np.maximum(0,X@self.W1+self.b1); z=a@self.W2+self.b2
        e=np.exp(z-z.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)
    def predict(self,X): return self.forward(X).argmax(1)
    def prob_class1(self,X): return self.forward(X)[:,1]
    def confidence(self,X): return self.forward(X).max(1)
    def train(self,X,y,epochs=1000,lr=0.01):
        yo=np.eye(2)[y]
        for _ in range(epochs):
            idx=np.random.choice(len(X),min(64,len(X)),replace=False)
            Xb,yb=X[idx],yo[idx]; p=self.forward(Xb); N=len(Xb)
            a=np.maximum(0,Xb@self.W1+self.b1); dz=(p-yb)/N
            self.W2-=lr*a.T@dz; self.b2-=lr*dz.sum(0,keepdims=True)
            da=dz@self.W2.T*(a>0); self.W1-=lr*Xb.T@da; self.b1-=lr*da.sum(0,keepdims=True)
    def accuracy(self,X,y): return np.mean(self.predict(X)==y)

FEATS_A = list(range(0,12)); FEATS_B = list(range(8,20))

def make_soft_xor(noise_frac, n):
    """
    y = XOR(A_sign, B_sign) but with `noise_frac` of labels flipped.
    noise_frac=0.0 → pure XOR (coalition ~0.50)
    noise_frac=0.3 → 70% XOR + 30% random → coalition gets some signal
    """
    X = np.random.randn(n, DIM)*0.3; X[:,FEATS_A]*=6; X[:,FEATS_B]*=6
    sig_a = (X[:,FEATS_A].sum(1)>0).astype(int)
    sig_b = (X[:,FEATS_B].sum(1)>0).astype(int)
    y = (sig_a ^ sig_b).astype(int)
    flip = np.random.choice(n, int(n*noise_frac), replace=False)
    y[flip] = 1 - y[flip]
    X += np.random.randn(*X.shape)*0.5
    return X, y

# Train solo experts
X_a = np.random.randn(300,DIM)*0.3; X_a[:,FEATS_A]*=6
y_a = ((X_a[:,FEATS_A].sum(1)>0)).astype(int); X_a+=np.random.randn(*X_a.shape)*0.5
X_b = np.random.randn(300,DIM)*0.3; X_b[:,FEATS_B]*=6
y_b = ((X_b[:,FEATS_B].sum(1)>0)).astype(int); X_b+=np.random.randn(*X_b.shape)*0.5

EA = Expert(42); EA.train(X_a,y_a)
EB = Expert(43); EB.train(X_b,y_b)

print("="*70)
print("ABLATION: Coalition Pseudo-Label — SOFT non-linearities")
print("="*70)
print(f"  Expert A solo acc: {EA.accuracy(X_a,y_a):.3f}")
print(f"  Expert B solo acc: {EB.accuracy(X_b,y_b):.3f}")
print(f"\n  {'Noise':>8} {'Coal avg':>10} {'Coal max':>10} {'Compiled':>10} {'Δ(Comp-Coal)':>15} {'Oracle':>10}")
print(f"  {'─'*65}")

for noise in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    X_pl, _ = make_soft_xor(noise, 600)
    X_test, y_test = make_soft_xor(noise, 400)

    # Coalition
    ca = np.mean(((EA.prob_class1(X_test)+EB.prob_class1(X_test))/2 > 0.5).astype(int) == y_test)
    cm = np.mean((np.maximum(EA.prob_class1(X_test),EB.prob_class1(X_test)) > 0.5).astype(int) == y_test)

    # Coalition → pseudo-labels → compiled
    pl = ((EA.prob_class1(X_pl)+EB.prob_class1(X_pl))/2 > 0.5).astype(int)
    C = Expert(99); C.train(X_pl, pl, epochs=1200)
    cc = C.accuracy(X_test, y_test)

    # Oracle
    O = Expert(77); O.train(X_test[:200], y_test[:200], epochs=1500)
    oa = O.accuracy(X_test[200:], y_test[200:])

    delta = cc - max(ca, cm)
    marker = " ←" if delta > 0.01 else ""
    print(f"  {noise:>8.1f} {ca:>10.3f} {cm:>10.3f} {cc:>10.3f} {delta:>+15.3f} {oa:>10.3f}{marker}")
