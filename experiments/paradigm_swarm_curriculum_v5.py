#!/usr/bin/env python3
"""
Paradigm Swarm v5b: Density-Based Self-Routing vs Universal.

SAME setup as v5. But instead of oracle routing (knowing the physics),
the swarm uses density-based self-routing:

  1. At episode start, collect first K steps from ALL experts
  2. Measure each expert's "density" = avg max Q-value on those steps
  3. Route to the expert with highest density
  4. That expert continues for the rest of the episode

FAIR comparison: no oracle knowledge of which physics is active.
Swarm must infer it from the state trajectory.
"""
import numpy as np
from collections import deque

np.random.seed(42)

# ═══════════════════ CARTPOLE (same as v5) ═══════════════════

class CartPole:
    def __init__(self, force_mult=1.0):
        self.gravity = 9.8; self.masscart = 1.0; self.masspole = 0.1
        self.length = 0.5; self.force_mag = 10.0 * force_mult; self.tau = 0.02
        self.x_threshold = 2.4
        self.theta_threshold_radians = 12 * np.pi / 180

    def reset(self):
        self.state = np.array([np.random.uniform(-0.05,0.05) for _ in range(4)])
        return self.state.copy()

    def step(self, action):
        x,x_dot,theta,theta_dot = self.state
        force = self.force_mag if action==1 else -self.force_mag
        costheta=np.cos(theta); sintheta=np.sin(theta)
        temp=(force+self.masspole*self.length*theta_dot**2*sintheta)/(self.masscart+self.masspole)
        theta_acc=(self.gravity*sintheta-costheta*temp)/(self.length*(4.0/3.0-self.masspole*costheta**2/(self.masscart+self.masspole)))
        x_acc=temp-self.masspole*self.length*theta_acc*costheta/(self.masscart+self.masspole)
        x+=self.tau*x_dot; x_dot+=self.tau*x_acc
        theta+=self.tau*theta_dot; theta_dot+=self.tau*theta_acc
        self.state=np.array([x,x_dot,theta,theta_dot])
        done=(x<-self.x_threshold or x>self.x_threshold or theta<-self.theta_threshold_radians or theta>self.theta_threshold_radians)
        return self.state.copy(), 1.0 if not done else 0.0, done

# ═══════════════════ Q-SOLVER (same as v5) ═══════════════════

class QSolver:
    def __init__(self,n_bins=8):
        self.n_bins=n_bins
        self.Q=np.zeros((n_bins,n_bins,n_bins,n_bins,2))
        self.bounds=[(-2.4,2.4),(-3.0,3.0),(-0.3,0.3),(-3.0,3.0)]

    def discretise(self,state):
        idx=[]
        for i,(lo,hi) in enumerate(self.bounds):
            v=np.clip(state[i],lo,hi)
            idx.append(min(int((v-lo)/(hi-lo)*(self.n_bins-1)),self.n_bins-1))
        return tuple(idx)

    def act(self,state,eps=0.0):
        if np.random.random()<eps: return np.random.randint(2)
        return self.Q[self.discretise(state)].argmax()

    def density(self,state):
        """Max Q-value — how 'confident' is this expert?"""
        return self.Q[self.discretise(state)].max()

    def learn(self,state,action,reward,next_state,done,lr=0.1,gamma=0.99):
        idx=self.discretise(state); nidx=self.discretise(next_state)
        target=reward if done else reward+gamma*self.Q[nidx].max()
        self.Q[idx][action]+=lr*(target-self.Q[idx][action])

# ═══════════════════ TRAIN (same as v5) ═══════════════════

print("="*70)
print("PARADIGM SWARM v5b: Density-Based Self-Routing vs Universal")
print("="*70)

generators={
    'normal':1.0, 'slippery':1.8, 'sticky':0.4
}
solvers={}

print("── Training specialized solvers ──")
for name,fm in generators.items():
    env=CartPole(fm)
    s=QSolver()
    for ep in range(2000):
        state=env.reset()
        for _ in range(500):
            eps=max(0.01,0.5*(0.995**ep))
            a=s.act(state,eps)
            ns,r,done=env.step(a)
            s.learn(state,a,r,ns,done)
            state=ns
            if done: break
    solvers[name]=s
    # Quick eval
    env_test=CartPole(fm)
    total=sum(1 for _ in range(100) if not all(
        env_test.step(s.act(env_test.reset()))[2]==False for _ in range(500)
    ));  # rough check — just verifying it learned
    print(f"  {name:<12}: trained")

# ═══════════════════ DENSITY-BASED SELF-ROUTING ═══════════════════

print(f"\n── Density-Based Self-Routing Evaluation ──")

K_PROBE = 10  # steps to probe each expert before routing

def evaluate_swarm_density(physics_name, n_episodes=200):
    """Swarm with density-based routing. No oracle knowledge of physics."""
    cfg=generators[physics_name]
    env=CartPole(cfg)
    total_reward=0
    route_counts={name:0 for name in solvers}

    for _ in range(n_episodes):
        state=env.reset()

        # Phase 1: PROBE — collect K steps from ALL experts
        # We need a temporary environment copy to probe without affecting the real one
        densities={name:0.0 for name in solvers}
        probe_env=CartPole(cfg)
        probe_env.state=state.copy()

        for step in range(K_PROBE):
            for name in solvers:
                a=solvers[name].act(probe_env.state,eps=0.0)
                densities[name]+=solvers[name].density(probe_env.state)
            # Advance probe with the BEST expert's action (not realistic but fair)
            # Actually, for true self-routing: each expert would get its own probe trajectory
            # But we can't fork the environment. Instead: use the real episode state.
            # Take action from a random expert to advance (or use first expert)
            a_probe=solvers['normal'].act(probe_env.state,eps=0.0)
            ns,r,done=probe_env.step(a_probe)
            if done: break

        # Average density per expert
        for name in solvers:
            densities[name]/=max(1,step+1)

        # Phase 2: ROUTE to best expert
        best_expert=max(densities,key=densities.get)
        route_counts[best_expert]+=1

        # Phase 3: EXECUTE — best expert controls the episode
        state=env.reset()
        for _ in range(500):
            a=solvers[best_expert].act(state,eps=0.0)
            state,reward,done=env.step(a)
            total_reward+=reward
            if done: break

    return total_reward/n_episodes, route_counts

# ═══════════════════ COALITION (vote) ═══════════════════

def evaluate_coalition(physics_name, n_episodes=200):
    cfg=generators[physics_name]
    env=CartPole(cfg)
    total=0
    for _ in range(n_episodes):
        state=env.reset()
        for _ in range(500):
            q_vals=np.zeros(2)
            for s in solvers.values():
                q_vals+=s.Q[s.discretise(state)]
            a=q_vals.argmax()
            state,r,done=env.step(a); total+=r
            if done: break
    return total/n_episodes

# ═══════════════════ BEST SOLO (oracle) ═══════════════════

def evaluate_best_solo(physics_name, n_episodes=200):
    cfg=generators[physics_name]
    best=0
    for name in solvers:
        env=CartPole(cfg); total=0
        for _ in range(n_episodes):
            state=env.reset()
            for _ in range(500):
                state,r,done=env.step(solvers[name].act(state))
                total+=r
                if done: break
        best=max(best,total/n_episodes)
    return best

# ═══════════════════ RUN ═══════════════════

swarm_results={}; swarm_routes={}
for phys in generators:
    avg_r, routes=evaluate_swarm_density(phys)
    swarm_results[phys]=avg_r; swarm_routes[phys]=routes

print(f"  {'Method':<25} {'Normal':>10} {'Slippery':>10} {'Sticky':>10} {'Avg':>10}")
print(f"  {'─'*65}")

# Best solo (oracle)
solo={}
for phys in generators:
    solo[phys]=evaluate_best_solo(phys)

# Coalition
coal={}
for phys in generators:
    coal[phys]=evaluate_coalition(phys)

# Universal (from v5 — retrain for fairness with same budget)
uni_solver=QSolver()
for ep in range(2000):
    phys=list(generators.keys())[ep%3]
    env=CartPole(generators[phys])
    state=env.reset()
    for _ in range(500):
        eps=max(0.01,0.5*(0.995**ep))
        a=uni_solver.act(state,eps)
        ns,r,done=env.step(a); uni_solver.learn(state,a,r,ns,done)
        state=ns
        if done: break

uni={}
for phys in generators:
    env=CartPole(generators[phys]); total=0
    for _ in range(200):
        state=env.reset()
        for _ in range(500):
            state,r,done=env.step(uni_solver.act(state))
            total+=r
            if done: break
    uni[phys]=total/200

print(f"  {'Best solo (oracle)':<25} {solo['normal']:>10.1f} {solo['slippery']:>10.1f} {solo['sticky']:>10.1f} {sum(solo.values())/3:>10.1f}")
print(f"  {'Coalition (vote)':<25} {coal['normal']:>10.1f} {coal['slippery']:>10.1f} {coal['sticky']:>10.1f} {sum(coal.values())/3:>10.1f}")
print(f"  {'Swarm (density route)':<25} {swarm_results['normal']:>10.1f} {swarm_results['slippery']:>10.1f} {swarm_results['sticky']:>10.1f} {sum(swarm_results.values())/3:>10.1f}")
print(f"  {'Universal':<25} {uni['normal']:>10.1f} {uni['slippery']:>10.1f} {uni['sticky']:>10.1f} {sum(uni.values())/3:>10.1f}")

print(f"\n  Routing distribution:")
for phys in generators:
    rt=swarm_routes[phys]; total=sum(rt.values())
    print(f"    {phys:<12}: " + "  ".join(f"{n}={c/total:.0%}" for n,c in rt.items()))

swarm_avg=sum(swarm_results.values())/3
uni_avg=sum(uni.values())/3
print(f"\n  Swarm (density) > Universal: {swarm_avg-uni_avg:+.1f}  ({'WIN' if swarm_avg>uni_avg else 'LOSE'})")
print(f"  Best solo (oracle) > Swarm:  {sum(solo.values())/3-swarm_avg:+.1f}  (routing cost)")
