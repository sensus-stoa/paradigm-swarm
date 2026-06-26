#!/usr/bin/env python3
"""
Curriculum Engine: Generator (map + obstacles) → Solver (pathfinding)

Generator: создаёт 2D-карту с препятствиями возрастающей сложности.
  Уровень сложности λ = obstacle_density.
  Если solver accuracy > 80% → λ растёт.
  Если accuracy < 50% → λ падает.
  Цель: найти границу «сложно но решаемо».

Solver: маленькая нейросеть (grid → direction). 
  Учится на демонстрациях A*.
  На каждом уровне λ: Generator создаёт 200 карт, A* решает их →
  обучающая выборка для Solver.
  Тест: 100 карт, Solver vs A*.

Архитектурно:
  G = генератор (rule-based, с параметром λ)
  S = решатель (neural net, учится)
  G получает feedback от S и адаптирует λ.
"""
import numpy as np
import heapq

np.random.seed(42)
GRID = 12  # 12×12 grid

# ─── A* Solver (optimal, used as teacher) ───
def astar(grid, start, goal):
    """A* pathfinding on binary grid. Returns path length or None if unreachable."""
    h, w = grid.shape
    def neighbors(r, c):
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0:
                yield nr, nc

    pq = [(0, start[0], start[1])]
    came_from = {}
    g_score = {start: 0}

    while pq:
        _, r, c = heapq.heappop(pq)
        if (r, c) == goal:
            # Reconstruct length
            return g_score[(r, c)]
        for nr, nc in neighbors(r, c):
            tentative = g_score[(r, c)] + 1
            if (nr, nc) not in g_score or tentative < g_score[(nr, nc)]:
                g_score[(nr, nc)] = tentative
                h_dist = abs(nr - goal[0]) + abs(nc - goal[1])
                heapq.heappush(pq, (tentative + h_dist, nr, nc))
                came_from[(nr, nc)] = (r, c)
    return None

def astar_direction(grid, start, goal):
    """Returns best first step direction as label: 0=up,1=down,2=left,3=right."""
    h, w = grid.shape
    pq = [(0, start[0], start[1], -1)]  # (priority, r, c, first_dir)
    came_from = {}
    g_score = {start: 0}
    first_step = {(start[0], start[1]): -1}

    while pq:
        _, r, c, first_d = heapq.heappop(pq)
        if (r, c) == goal:
            return first_d
        for d, (dr, dc) in enumerate([(-1,0),(1,0),(0,-1),(0,1)]):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0:
                tentative = g_score[(r, c)] + 1
                if (nr, nc) not in g_score or tentative < g_score[(nr, nc)]:
                    g_score[(nr, nc)] = tentative
                    fd = first_d if first_d >= 0 else d
                    first_step[(nr, nc)] = fd
                    h_dist = abs(nr - goal[0]) + abs(nc - goal[1])
                    heapq.heappush(pq, (tentative + h_dist, nr, nc, fd))
    return -1  # unreachable

# ─── Generator ───
def generate_map(density, size=GRID):
    """Grid with random obstacles at given density. Start at (0,0), goal at (size-1,size-1)."""
    grid = np.zeros((size, size), dtype=int)
    n_obstacles = int(size * size * density)
    positions = np.random.choice(size * size - 2, n_obstacles, replace=False)
    for pos in positions:
        r, c = pos // size, pos % size
        if (r, c) != (0, 0) and (r, c) != (size - 1, size - 1):
            grid[r, c] = 1
    return grid

def generate_dataset(density, n_samples, size=GRID):
    """Generate n_samples maps with A* direction labels. Returns (X, y) or None if too few solvable."""
    X = np.zeros((n_samples, size, size), dtype=np.float32)
    y = np.full(n_samples, -1, dtype=int)
    start = (0, 0)
    goal = (size - 1, size - 1)

    solved = 0
    for i in range(n_samples):
        grid = generate_map(density, size)
        direction = astar_direction(grid, start, goal)
        if direction >= 0:
            X[solved] = grid
            y[solved] = direction
            solved += 1

    if solved < 10:
        return None, 0
    return X[:solved], y[:solved], solved

# ─── Solver (neural net) ───
class SolverNN:
    def __init__(self):
        rng = np.random.RandomState(42); s = 0.05
        n_in = GRID * GRID
        self.W1 = rng.randn(n_in, 64) * s; self.b1 = np.zeros(64)
        self.W2 = rng.randn(64, 4) * s; self.b2 = np.zeros(4)

    def forward(self, X_flat):
        a = np.maximum(0, X_flat @ self.W1 + self.b1)
        z = a @ self.W2 + self.b2
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def predict(self, X_flat):
        return self.forward(X_flat).argmax(1)

    def accuracy(self, X_flat, y):
        return np.mean(self.predict(X_flat) == y)

    def train(self, X_flat, y, epochs=200, lr=0.01):
        yo = np.eye(4)[y]
        for _ in range(epochs):
            idx = np.random.choice(len(X_flat), min(128, len(X_flat)), replace=False)
            Xb, yb = X_flat[idx], yo[idx]
            p = self.forward(Xb); N = len(Xb)
            a = np.maximum(0, Xb @ self.W1 + self.b1)
            dz = (p - yb) / N
            self.W2 -= lr * a.T @ dz; self.b2 -= lr * dz.sum(0)
            da = dz @ self.W2.T * (a > 0)
            self.W1 -= lr * Xb.T @ da; self.b1 -= lr * da.sum(0)

# ─── Curriculum Loop ───
print("=" * 70)
print("CURRICULUM ENGINE: Generator → Solver co-evolution")
print("=" * 70)

solver = SolverNN()
density = 0.05   # start easy
history = []
density_history = []

for epoch in range(20):
    # Generator: create dataset at current density
    X, y, solved = generate_dataset(density, 300)
    if X is None:
        print(f"  Epoch {epoch:2d}: density={density:.2f}  NO solvable maps → back off")
        density *= 0.7
        continue

    # Solver: train on this data
    X_flat = X.reshape(solved, -1)
    solver.train(X_flat, y, epochs=150)

    # Test on fresh maps at same density
    X_test, y_test, solved_test = generate_dataset(density, 100)
    if X_test is not None:
        Xt_flat = X_test.reshape(solved_test, -1)
        acc = solver.accuracy(Xt_flat, y_test)
    else:
        acc = 0.0

    solvable_rate = solved / 300  # how many maps A* can solve

    history.append((epoch, density, acc, solvable_rate))
    density_history.append(density)
    print(f"  E{epoch:2d}: λ={density:.2f}  solvable={solvable_rate:.0%}  "
          f"solver_acc={acc:.3f}  samples={solved}")

    # Curriculum: adjust density based on solver performance
    if acc > 0.80 and solvable_rate > 0.5:
        density = min(0.45, density + 0.03)  # harder
    elif acc < 0.50 or solvable_rate < 0.2:
        density = max(0.05, density - 0.03)  # easier
    # else: stay

# ─── Visualise curriculum ───
print(f"\n{'='*70}")
print(f"CURRICULUM PATH")
print(f"{'='*70}")
print(f"  {'Epoch':<8} {'Density':>8} {'Solvable':>10} {'SolverAcc':>10}")
print(f"  {'─'*40}")
for ep, d, sr, acc in history:
    bar = '█' * int(d * 40) + '░' * (18 - int(d * 40))
    print(f"  {ep:<8} {d:>8.2f} {sr:>10.0%} {acc:>10.3f}  {bar}")

print(f"\n  Final density: {density:.2f}")
print(f"  Density range: {min(density_history):.2f} → {max(density_history):.2f}")

# ─── Test solver on range of densities ───
print(f"\n{'='*70}")
print(f"SOLVER GENERALISATION: accuracy vs obstacle density")
print(f"{'='*70}")
for d in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    Xt, yt, st = generate_dataset(d, 100)
    if Xt is not None:
        acc = solver.accuracy(Xt.reshape(st, -1), yt)
        bar = '█' * int(acc * 20) + '░' * (20 - int(acc * 20))
        print(f"  λ={d:.2f}: solv={st:3d}  acc={acc:.3f}  {bar}")
    else:
        print(f"  λ={d:.2f}: NO solvable maps")
