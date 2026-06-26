#!/usr/bin/env python3
"""
Generator Swarm → Solver Swarm  v4b: RL on Multi-Surface.

SAME grid, DIFFERENT dynamics. Solvers learn via Q-learning on actual surface.
Normal: deterministic. Slippery: 25% slide. Sticky: 25% no-move.
Different dynamics → different optimal strategies → measurable specialization.
"""
import numpy as np
from collections import deque

np.random.seed(42)
GRID = 8  # smaller for faster RL
ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]

def make_grid(density):
    g=np.zeros((GRID,GRID),dtype=int)
    n=int(GRID*GRID*density)
    pos=np.random.choice(GRID*GRID-2,n,replace=False)
    for p in pos:
        r,c=p//GRID,p%GRID
        if (r,c) not in [(0,0),(GRID-1,GRID-1)]: g[r,c]=1
    return g

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

def generate_solvable_grids(n, density=0.12):
    grids=[]
    for _ in range(n*3):  # over-generate
        g=make_grid(density)
        if path_length_bfs(g)>0: grids.append(g)
        if len(grids)>=n: break
    return grids

# ═══════════════════ SURFACE SIMULATORS ═══════════════════

def step_normal(grid, r, c, action):
    dr,dc=ACTIONS[action]
    nr,nc=r+dr,c+dc
    if 0<=nr<GRID and 0<=nc<GRID and grid[nr,nc]==0: return nr,nc
    return r,c  # bump wall

def step_slippery(grid, r, c, action):
    dr,dc=ACTIONS[action]
    nr,nc=r+dr,c+dc
    if 0<=nr<GRID and 0<=nc<GRID and grid[nr,nc]==0:
        if np.random.random()<0.25:  # slide
            nr2,nc2=nr+dr,nc+dc
            if 0<=nr2<GRID and 0<=nc2<GRID and grid[nr2,nc2]==0:
                return nr2,nc2
        return nr,nc
    return r,c

def step_sticky(grid, r, c, action):
    if np.random.random()<0.25: return r,c  # stuck
    dr,dc=ACTIONS[action]
    nr,nc=r+dr,c+dc
    if 0<=nr<GRID and 0<=nc<GRID and grid[nr,nc]==0: return nr,nc
    return r,c

# ═══════════════════ Q-LEARNING SOLVER ═══════════════════
class QSolver:
    def __init__(self,seed=42):
        rng=np.random.RandomState(seed); s=0.05; n_in=GRID*GRID
        self.W1=rng.randn(n_in,128)*s; self.b1=np.zeros(128)
        self.W2=rng.randn(128,4)*s; self.b2=np.zeros(4)
        self.N_ACTIONS = len(ACTIONS)
    def q_values(self,grid_flat):
        a=np.maximum(0,grid_flat@self.W1+self.b1)
        return a@self.W2+self.b2
    def act(self,gf,eps=0.1):
        if np.random.random()<eps: return np.random.randint(self.N_ACTIONS)
        return self.q_values(gf).argmax()
    def train_step(self,states,actions,rewards,next_states,dones,lr=0.002,gamma=0.95):
        n=len(states); q=self.q_values(states); qn=self.q_values(next_states)
        targets=q.copy()
        for i in range(n):
            targets[i,actions[i]] = rewards[i] if dones[i] else rewards[i]+gamma*qn[i].max()
        a1=np.maximum(0,states@self.W1+self.b1)
        dz=(q-targets)/n
        self.W2-=lr*a1.T@dz; self.b2-=lr*dz.mean(0)
        da=dz@self.W2.T*(a1>0)
        self.W1-=lr*states.T@da; self.b1-=lr*da.mean(0)

def run_episode(solver, grid, step_fn, max_steps=200, eps=0.1, train=True):
    r,c=0,0
    states,actions,rewards,next_states,dones=[],[],[],[],[]
    for _ in range(max_steps):
        s=grid.flatten().astype(np.float32)
        a=solver.act(s,eps)
        nr,nc=step_fn(grid,r,c,a)
        done=False; reward=-0.01
        if nr==r and nc==c: reward=-0.05  # bump/stuck penalty
        if (nr,nc)==(GRID-1,GRID-1): reward=1.0; done=True
        ns=grid.flatten().astype(np.float32)
        states.append(s); actions.append(a); rewards.append(reward)
        next_states.append(ns); dones.append(done)
        r,c=nr,nc
        if done: break
    if train and len(states)>0:
        solver.train_step(np.array(states),np.array(actions),
                         np.array(rewards),np.array(next_states),np.array(dones))
    return 1 if (r,c)==(GRID-1,GRID-1) else 0

def evaluate(solver, grids, step_fn, eps=0.0):
    return sum(run_episode(solver,g,step_fn,eps=eps,train=False) for g in grids)

# ═══════════════════ MAIN ═══════════════════
print("="*70)
print("GENERATOR SWARM → SOLVER SWARM  v4b: RL Multi-Surface")
print("="*70)

# Shared grid set (SAME grids, DIFFERENT surfaces)
train_grids=generate_solvable_grids(60,density=0.12)
test_grids=generate_solvable_grids(40,density=0.12)
print(f"  Grids: {len(train_grids)} train, {len(test_grids)} test")

# Train specialized solvers
SN=QSolver(42); SS=QSolver(43); ST=QSolver(44)

print("\n── Training specialized solvers ──")
for epoch in range(40):
    for g in train_grids:
        eps=max(0.05,0.4-epoch*0.01)
        run_episode(SN,g,step_normal,eps=eps,max_steps=100)
        run_episode(SS,g,step_slippery,eps=eps,max_steps=100)
        run_episode(ST,g,step_sticky,eps=eps,max_steps=100)

    if epoch%20==0:
        sn=evaluate(SN,test_grids[:40],step_normal)
        ss=evaluate(SS,test_grids[:40],step_slippery)
        st=evaluate(ST,test_grids[:40],step_sticky)
        print(f"  e{epoch:3d}: SN={sn:2d}  SS={ss:2d}  ST={st:2d}")

# ═══════════════════ CROSS-SURFACE ═══════════════════
print(f"\n{'='*70}")
print(f"CROSS-SURFACE: Specialization (RL)")
print(f"{'='*70}")

sn_n=evaluate(SN,test_grids,step_normal)
sn_s=evaluate(SN,test_grids,step_slippery)
sn_t=evaluate(SN,test_grids,step_sticky)

ss_n=evaluate(SS,test_grids,step_normal)
ss_s=evaluate(SS,test_grids,step_slippery)
ss_t=evaluate(SS,test_grids,step_sticky)

st_n=evaluate(ST,test_grids,step_normal)
st_s=evaluate(ST,test_grids,step_slippery)
st_t=evaluate(ST,test_grids,step_sticky)

print(f"  {'Solver':<15} {'Normal':>10} {'Slippery':>10} {'Sticky':>10}")
print(f"  {'─'*45}")
print(f"  {'SN (normal)':<15} {sn_n:>10} {sn_s:>10} {sn_t:>10}")
print(f"  {'SS (slippery)':<15} {ss_n:>10} {ss_s:>10} {ss_t:>10}")
print(f"  {'ST (sticky)':<15} {st_n:>10} {st_s:>10} {st_t:>10}")

spec_n = sn_n - (sn_s+sn_t)/2
spec_s = ss_s - (ss_n+ss_t)/2
spec_t = st_t - (st_n+st_s)/2
print(f"\n  Specialization (own - cross avg):")
print(f"    SN: {spec_n:+.1f}  SS: {spec_s:+.1f}  ST: {spec_t:+.1f}")
print(f"    {'✓ DIFFERENT!' if max(abs(spec_n),abs(spec_s),abs(spec_t))>2 else '⚠ similar'}")

# Universal solver
SU=QSolver(45)
print(f"\n── Training universal solver ──")
for epoch in range(40):
    for g in train_grids:
        eps=max(0.05,0.4-epoch*0.01)
        run_episode(SU,g,step_normal,eps=eps,max_steps=100)
        run_episode(SU,g,step_slippery,eps=eps,max_steps=100)
        run_episode(SU,g,step_sticky,eps=eps,max_steps=100)
    if epoch%20==0:
        su=evaluate(SU,test_grids[:40],step_normal)+evaluate(SU,test_grids[:40],step_slippery)+evaluate(SU,test_grids[:40],step_sticky)
        print(f"  e{epoch:3d}: SU(total)={su}")

su_n=evaluate(SU,test_grids,step_normal)
su_s=evaluate(SU,test_grids,step_slippery)
su_t=evaluate(SU,test_grids,step_sticky)
print(f"\n  Universal: N={su_n}  S={su_s}  T={su_t}  avg={(su_n+su_s+su_t)/3:.1f}")

best_spec_avg = max((sn_n+ss_s+st_t)/3, (sn_n+sn_s+sn_t)/3, (ss_n+ss_s+ss_t)/3, (st_n+st_s+st_t)/3)
univ_avg = (su_n+su_s+su_t)/3
print(f"\n  Best specialized avg: {best_spec_avg:.1f}")
print(f"  Universal avg:        {univ_avg:.1f}")
print(f"  Swarm advantage:      {best_spec_avg-univ_avg:+.1f}")
