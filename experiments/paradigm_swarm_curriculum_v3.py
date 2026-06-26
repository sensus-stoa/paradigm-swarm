#!/usr/bin/env python3
"""
Generator Swarm → Solver Swarm  v3c: A*-pretrained solvers.

Solvers pretrained on A* demonstrations → can navigate successfully.
Then: different generators → different training distributions → specialization?
Cross-type test: SA on GB, SB on GA.
Coalition vs Compiled on GC (mixed).

Fast — supervised learning, no RL exploration bottleneck.
"""
import numpy as np
from collections import deque

np.random.seed(42)
GRID = 14
ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]

def path_length_bfs(grid):
    q=deque([(0,0,0)]); visited={(0,0)}
    while q:
        r,c,d=q.popleft()
        if (r,c)==(GRID-1,GRID-1): return d
        for dr,dc in ACTIONS:
            nr,nc=r+dr,c+dc
            if 0<=nr<GRID and 0<=nc<GRID and grid[nr,nc]==0 and (nr,nc) not in visited:
                visited.add((nr,nc)); q.append((nr,nc,d+1))
    return -1

def astar_direction(grid, start=(0,0), goal=None):
    if goal is None: goal=(GRID-1,GRID-1)
    import heapq
    h,w=grid.shape
    pq=[(0,start[0],start[1],-1)]
    g_score={start:0}; first_step={start:-1}
    while pq:
        _,r,c,fd=heapq.heappop(pq)
        if (r,c)==goal: return fd
        for d,(dr,dc) in enumerate(ACTIONS):
            nr,nc=r+dr,c+dc
            if 0<=nr<h and 0<=nc<w and grid[nr,nc]==0:
                tg=g_score[(r,c)]+1
                if (nr,nc) not in g_score or tg<g_score[(nr,nc)]:
                    g_score[(nr,nc)]=tg
                    first_step[(nr,nc)]=fd if fd>=0 else d
                    h_dist=abs(nr-goal[0])+abs(nc-goal[1])
                    heapq.heappush(pq,(tg+h_dist,nr,nc,first_step[(nr,nc)]))
    return -1

# ═══════════════════ GENERATOR ═══════════════════
def make_scattered(density, size=GRID):
    g=np.zeros((size,size),dtype=int)
    n=int(size*size*density)
    pos=np.random.choice(size*size-2,n,replace=False)
    for p in pos:
        r,c=p//size,p%size
        if (r,c) not in [(0,0),(size-1,size-1)]: g[r,c]=1
    return g

def make_corridor(n_bars, size=GRID):
    g=np.zeros((size,size),dtype=int)
    for _ in range(n_bars):
        if np.random.random()<0.5:
            col=np.random.randint(2,size-2)
            gaps=sorted(np.random.choice(range(1,size-1),max(1,size//5),replace=False))
            g[:,col]=1
            for gap in gaps: g[gap,col]=0
        else:
            row=np.random.randint(2,size-2)
            gaps=sorted(np.random.choice(range(1,size-1),max(1,size//5),replace=False))
            g[row,:]=1
            for gap in gaps: g[row,gap]=0
    g[0,0]=0; g[size-1,size-1]=0
    return g

def make_mixed(wall_d, n_bars, size=GRID):
    g=np.zeros((size,size),dtype=int)
    nw=int(size*size*wall_d)
    pos=np.random.choice(size*size-2,nw,replace=False)
    for p in pos:
        r,c=p//size,p%size
        if (r,c) not in [(0,0),(size-1,size-1)]: g[r,c]=1
    for _ in range(n_bars):
        if np.random.random()<0.5:
            col=np.random.randint(2,size-2)
            gaps=sorted(np.random.choice(range(1,size-1),2,replace=False))
            g[:,col]=1
            for gap in gaps: g[gap,col]=0
        else:
            row=np.random.randint(2,size-2)
            gaps=sorted(np.random.choice(range(1,size-1),2,replace=False))
            g[row,:]=1
            for gap in gaps: g[row,gap]=0
    g[0,0]=0; g[size-1,size-1]=0
    return g

def gen_dataset(gen_fn, n_samples):
    X=np.zeros((n_samples,GRID*GRID),dtype=np.float32)
    y=np.full(n_samples,-1,dtype=int); s=0
    for i in range(n_samples):
        g=gen_fn(); d=astar_direction(g)
        if d>=0: X[s]=g.flatten(); y[s]=d; s+=1
    if s<15: return None,None,0
    return X[:s],y[:s],s

# ═══════════════════ SOLVER ═══════════════════
class SolverNN:
    def __init__(self,seed=42):
        rng=np.random.RandomState(seed); s=0.05; n_in=GRID*GRID
        self.W1=rng.randn(n_in,128)*s; self.b1=np.zeros(128)
        self.W2=rng.randn(128,4)*s; self.b2=np.zeros(4)
    def forward(self,X):
        a=np.maximum(0,X@self.W1+self.b1); z=a@self.W2+self.b2
        e=np.exp(z-z.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)
    def predict(self,X): return self.forward(X).argmax(1)
    def prob(self,X): return self.forward(X)
    def accuracy(self,X,y): return np.mean(self.predict(X)==y)
    def train(self,X,y,epochs=200,lr=0.01):
        yo=np.eye(4)[y]
        for _ in range(epochs):
            idx=np.random.choice(len(X),min(128,len(X)),replace=False)
            Xb,yb=X[idx],yo[idx]; p=self.forward(Xb); N=len(Xb)
            a=np.maximum(0,Xb@self.W1+self.b1); dz=(p-yb)/N
            self.W2-=lr*a.T@dz; self.b2-=lr*dz.sum(0)
            da=dz@self.W2.T*(a>0); self.W1-=lr*Xb.T@da; self.b1-=lr*da.sum(0)

# ═══════════════════ MAIN ═══════════════════
print("="*70)
print(f"GENERATOR SWARM → SOLVER SWARM  v3c (A*-pretrained, {GRID}x{GRID})")
print("="*70)

# Generate training data from each generator
X_a,y_a,sa=gen_dataset(lambda:make_scattered(0.20),800)
X_b,y_b,sb=gen_dataset(lambda:make_corridor(3),800)
X_c,y_c,sc=gen_dataset(lambda:make_mixed(0.12,2),800)
print(f"  GA scattered: {sa} solvable  GB corridor: {sb}  GC mixed: {sc}")

# Train specialized solvers
SA=SolverNN(42); SA.train(X_a,y_a,epochs=300)
SB=SolverNN(43); SB.train(X_b,y_b,epochs=300)
print(f"  SA→GA: {SA.accuracy(X_a,y_a):.3f}  SB→GB: {SB.accuracy(X_b,y_b):.3f}")

# Cross-type test
X_at,y_at,_=gen_dataset(lambda:make_scattered(0.20),200)
X_bt,y_bt,_=gen_dataset(lambda:make_corridor(3),200)
X_ct,y_ct,_=gen_dataset(lambda:make_mixed(0.12,2),200)

sa_a=SA.accuracy(X_at,y_at) if X_at is not None else 0
sa_b=SA.accuracy(X_bt,y_bt) if X_bt is not None else 0
sb_b=SB.accuracy(X_bt,y_bt) if X_bt is not None else 0
sb_a=SB.accuracy(X_at,y_at) if X_at is not None else 0

print(f"\n  Cross-type:")
print(f"    SA on GA (own):      {sa_a:.3f}")
print(f"    SA on GB (cross):    {sa_b:.3f}  Δ={sa_a-sa_b:+.3f}")
print(f"    SB on GB (own):      {sb_b:.3f}")
print(f"    SB on GA (cross):    {sb_a:.3f}  Δ={sb_b-sb_a:+.3f}")
spec = abs(sa_a-sa_b) + abs(sb_b-sb_a)
print(f"    Specialization: {spec:.3f}  {'✓ DIFFERENT!' if spec>0.05 else '⚠ no specialization'}")

# Coalition vs Compiled on Mixed
if X_ct is not None and y_ct is not None:
    # Coalition: soft vote
    p_a=SA.prob(X_ct); p_b=SB.prob(X_ct)
    coal_pred=(p_a+p_b).argmax(1)
    coal_acc=np.mean(coal_pred==y_ct)

    # Compiled: trained on BOTH GA and GB data
    X_ab=np.vstack([X_a,X_b]); y_ab=np.hstack([y_a,y_b])
    SC=SolverNN(99); SC.train(X_ab,y_ab,epochs=400)
    comp_acc=SC.accuracy(X_ct,y_ct)

    best_solo=max(sa_a,sb_b,
                  SA.accuracy(X_ct,y_ct) if X_ct is not None else 0,
                  SB.accuracy(X_ct,y_ct) if X_ct is not None else 0)

    print(f"\n  On MIXED (GC):")
    print(f"    Best Solo:     {best_solo:.3f}")
    print(f"    Coalition A+B: {coal_acc:.3f}")
    print(f"    Compiled AB:   {comp_acc:.3f}")
    print(f"    Winner:        {'Compiled' if comp_acc>max(best_solo,coal_acc) else ('Coalition' if coal_acc>best_solo else 'Solo')}")
