#!/usr/bin/env python3
"""
Curriculum Engine v2: Generator Swarm → Solver Swarm  (GRID=20)
Three distinct obstacle types on 20x20 grid.
Tests: solo vs coalition vs compiled on mixed obstacles.
"""
import numpy as np
import heapq
np.random.seed(42)

GRID = 20

def astar_direction(grid, start=(0,0), goal=None):
    if goal is None: goal = (GRID-1, GRID-1)
    h, w = grid.shape
    pq = [(0, start[0], start[1], -1)]
    g_score = {start: 0}; first_step = {start: -1}
    while pq:
        _, r, c, fd = heapq.heappop(pq)
        if (r,c) == goal: return fd
        for d, (dr,dc) in enumerate([(-1,0),(1,0),(0,-1),(0,1)]):
            nr, nc = r+dr, c+dc
            if 0<=nr<h and 0<=nc<w and grid[nr,nc]==0:
                tg = g_score[(r,c)]+1
                if (nr,nc) not in g_score or tg < g_score[(nr,nc)]:
                    g_score[(nr,nc)] = tg
                    first_step[(nr,nc)] = fd if fd>=0 else d
                    h_dist = abs(nr-goal[0])+abs(nc-goal[1])
                    heapq.heappush(pq, (tg+h_dist, nr, nc, first_step[(nr,nc)]))
    return -1

# ═══════════════════ GENERATORS ═══════════════════

def gen_walls(density):
    """Scattered block obstacles. Creates wide detours."""
    g = np.zeros((GRID,GRID), dtype=int)
    n = int(GRID*GRID*density)
    pos = np.random.choice(GRID*GRID-2, n, replace=False)
    for p in pos:
        r, c = p//GRID, p%GRID
        if (r,c) not in [(0,0),(GRID-1,GRID-1)]:
            g[r,c] = 1
    return g

def gen_narrow(n_bars):
    """Long horizontal/vertical bars with small gaps. Forces specific corridors."""
    g = np.zeros((GRID,GRID), dtype=int)
    for _ in range(n_bars):
        if np.random.random() < 0.5:
            col = np.random.randint(2, GRID-2)
            gaps = sorted(np.random.choice(range(1,GRID-1), 2, replace=False))
            g[:, col] = 1
            for gap in gaps: g[gap, col] = 0
        else:
            row = np.random.randint(2, GRID-2)
            gaps = sorted(np.random.choice(range(1,GRID-1), 2, replace=False))
            g[row, :] = 1
            for gap in gaps: g[row, gap] = 0
    return g

def gen_dynamic(n_patches):
    """Small random block clusters. Unpredictable but passable."""
    g = np.zeros((GRID,GRID), dtype=int)
    for _ in range(n_patches):
        r, c = np.random.randint(1,GRID-2), np.random.randint(1,GRID-2)
        w, h = np.random.randint(1,4), np.random.randint(1,4)
        g[r:min(r+h,GRID-1), c:min(c+w,GRID-1)] = 1
    g[0,0] = 0; g[GRID-1,GRID-1] = 0
    return g

def gen_mixed(wall_d, n_bars, n_patches):
    g = np.zeros((GRID,GRID), dtype=int)
    nw = int(GRID*GRID*wall_d)
    pos = np.random.choice(GRID*GRID-2, nw, replace=False)
    for p in pos:
        r, c = p//GRID, p%GRID
        if (r,c) not in [(0,0),(GRID-1,GRID-1)]: g[r,c] = 1
    for _ in range(n_bars):
        if np.random.random()<0.5:
            col=np.random.randint(2,GRID-2); gaps=sorted(np.random.choice(range(1,GRID-1),2,replace=False))
            g[:,col]=1
            for gap in gaps: g[gap,col]=0
        else:
            row=np.random.randint(2,GRID-2); gaps=sorted(np.random.choice(range(1,GRID-1),2,replace=False))
            g[row,:]=1
            for gap in gaps: g[row,gap]=0
    for _ in range(n_patches):
        r,c=np.random.randint(1,GRID-2),np.random.randint(1,GRID-2); w,h=np.random.randint(1,3),np.random.randint(1,3)
        g[r:min(r+h,GRID-1),c:min(c+w,GRID-1)]=1
    g[0,0]=0; g[GRID-1,GRID-1]=0
    return g

# ═══════════════════ SOLVER ═══════════════════

class SolverNN:
    def __init__(self, seed=42):
        rng=np.random.RandomState(seed); s=0.05
        n_in=GRID*GRID
        self.W1=rng.randn(n_in,128)*s; self.b1=np.zeros(128)
        self.W2=rng.randn(128,4)*s; self.b2=np.zeros(4)
    def forward(self,X):
        a=np.maximum(0,X@self.W1+self.b1); z=a@self.W2+self.b2
        e=np.exp(z-z.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)
    def predict(self,X): return self.forward(X).argmax(1)
    def prob(self,X): return self.forward(X)
    def accuracy(self,X,y): return np.mean(self.predict(X)==y)
    def train(self,X,y,epochs=300,lr=0.01):
        yo=np.eye(4)[y]
        for _ in range(epochs):
            idx=np.random.choice(len(X),min(128,len(X)),replace=False)
            Xb,yb=X[idx],yo[idx]; p=self.forward(Xb); N=len(Xb)
            a=np.maximum(0,Xb@self.W1+self.b1); dz=(p-yb)/N
            self.W2-=lr*a.T@dz; self.b2-=lr*dz.sum(0)
            da=dz@self.W2.T*(a>0); self.W1-=lr*Xb.T@da; self.b1-=lr*da.sum(0)

def gen_dataset(gen_fn, n):
    X=np.zeros((n,GRID*GRID),dtype=np.float32); y=np.full(n,-1,dtype=int); s=0
    for i in range(n):
        g=gen_fn(); d=astar_direction(g)
        if d>=0: X[s]=g.flatten(); y[s]=d; s+=1
    if s<15: return None,None,0
    return X[:s],y[:s],s

# ═══════════════════ TRAIN ═══════════════════
print("="*70)
print(f"GENERATOR SWARM → SOLVER SWARM  ({GRID}x{GRID})")
print("="*70)

Xw,yw,sw=gen_dataset(lambda:gen_walls(0.20),800)
Sw=SolverNN(42); Sw.train(Xw,yw,epochs=400)
print(f"  S_walls (d=0.20):   {sw} solvable, acc={Sw.accuracy(Xw,yw):.3f}")

Xn,yn,sn=gen_dataset(lambda:gen_narrow(4),800)
Sn=SolverNN(43); Sn.train(Xn,yn,epochs=400)
print(f"  S_narrow (bars=4):  {sn} solvable, acc={Sn.accuracy(Xn,yn):.3f}")

Xd,yd,sd=gen_dataset(lambda:gen_dynamic(20),800)
Sd=SolverNN(44); Sd.train(Xd,yd,epochs=400)
print(f"  S_dynamic (patches=20): {sd} solvable, acc={Sd.accuracy(Xd,yd):.3f}")

# Compiled W+N
Xwn,ywn,swn=gen_dataset(lambda:gen_mixed(0.12,3,10),800)
Swn=SolverNN(99); Swn.train(Xwn,ywn,epochs=500)
print(f"  Compiled W+N:       {swn} solvable, acc={Swn.accuracy(Xwn,ywn):.3f}")

# ═══════════════════ TEST ACROSS DIFFICULTIES ═══════════════════
print(f"\n{'='*70}")
print(f"FINAL: Solo vs Coalition vs Compiled across difficulties")
print(f"{'='*70}")
print(f"  {'Level':<20} {'Solvable':>8} {'Best Solo':>10} {'Coalition':>10} {'Compiled':>10} {'Winner':>10}")
print(f"  {'─'*70}")

levels = [
    ("Easy",      0.08, 2, 6),
    ("Medium",    0.12, 3, 10),
    ("Hard",      0.16, 4, 14),
    ("Very Hard", 0.20, 5, 18),
    ("Extreme",   0.24, 6, 22),
]

for label, wd, nb, np_ in levels:
    Xt,yt,st=gen_dataset(lambda:gen_mixed(wd,nb,np_),300)
    if Xt is None or st<20:
        print(f"  {label:<20} {st:>8} {'—':>10} {'—':>10} {'—':>10}  (unsolvable)")
        continue

    solo_acc=max(Sw.accuracy(Xt,yt),Sn.accuracy(Xt,yt),Sd.accuracy(Xt,yt))
    pw=Sw.prob(Xt); pn=Sn.prob(Xt); pd=Sd.prob(Xt)
    coal_acc=np.mean(((pw+pn+pd)/3).argmax(1)==yt)
    comp_acc=Swn.accuracy(Xt,yt)

    winner = "Compiled" if comp_acc>max(solo_acc,coal_acc) else ("Coalition" if coal_acc>solo_acc else "Solo")
    print(f"  {label:<20} {st:>8} {solo_acc:>10.3f} {coal_acc:>10.3f} {comp_acc:>10.3f}  {winner}")

# ═══════════════════ CROSS-TYPE GENERALISATION ═══════════════════
print(f"\n{'='*70}")
print(f"CROSS-TYPE: How well do solvers generalise to OTHER obstacle types?")
print(f"{'='*70}")
print(f"  {'Solver':<15} {'Walls':>10} {'Narrow':>10} {'Dynamic':>10} {'Mixed':>10}")
print(f"  {'─'*58}")

Xtw,ytw,_=gen_dataset(lambda:gen_walls(0.15),200)
Xtn,ytn,_=gen_dataset(lambda:gen_narrow(3),200)
Xtd,ytd,_=gen_dataset(lambda:gen_dynamic(15),200)
Xtm,ytm,_=gen_dataset(lambda:gen_mixed(0.10,2,8),200)

for name,solver in [("S_walls",Sw),("S_narrow",Sn),("S_dynamic",Sd),("Compiled W+N",Swn)]:
    if Xtw is not None: aw=solver.accuracy(Xtw,ytw) if Xtw is not None else 0
    else: aw=0
    an=solver.accuracy(Xtn,ytn) if Xtn is not None else 0
    ad=solver.accuracy(Xtd,ytd) if Xtd is not None else 0
    am=solver.accuracy(Xtm,ytm) if Xtm is not None else 0
    print(f"  {name:<15} {aw:>10.3f} {an:>10.3f} {ad:>10.3f} {am:>10.3f}")
