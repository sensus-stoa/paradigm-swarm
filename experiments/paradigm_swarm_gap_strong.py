#!/usr/bin/env python3
"""Paradigm Swarm — Strong Gap Detection via Gaussian Densities
Each paradigm = multivariate Gaussian (mean + covariance).
Gap = point where density < threshold for ALL paradigms.
Monolithic softmax FORCES a choice. PS naturally says 'none'.
"""
import numpy as np

np.random.seed(42)

# 3 well-separated clusters
N = 150
centers = {
    "Labour": np.array([-5, -5]),
    "Thermo": np.array([5, -5]),
    "Geo": np.array([0, 6]),
}
X, y, names = [], [], []
for i, (name, c) in enumerate(centers.items()):
    pts = np.random.randn(N, 2) * 1.0 + c
    X.append(pts); y.append(np.full(N, i)); names.append(name)
X = np.vstack(X); y = np.hstack(y)
y_oh = np.eye(3)[y]

# ═══════════════════════════════════════════════════
# MONOLITHIC: multi-class softmax (forces choice)
# ═══════════════════════════════════════════════════
class MonoMLP:
    def __init__(self):
        rng = np.random.RandomState(42)
        self.W1 = rng.randn(2,16)*0.5; self.b1 = np.zeros((1,16))
        self.W2 = rng.randn(16,3)*0.5; self.b2 = np.zeros((1,3))
    def forward(self, X):
        a1 = np.tanh(X @ self.W1 + self.b1)
        z = a1 @ self.W2 + self.b2
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
    def train(self, X, y_oh, epochs=500, lr=0.05):
        for _ in range(epochs):
            p = self.forward(X); N = X.shape[0]
            dz = (p - y_oh) / N
            a1 = np.tanh(X @ self.W1 + self.b1)
            self.W2 -= lr * a1.T @ dz; self.b2 -= lr * dz.sum(0, keepdims=True)
            da = dz @ self.W2.T
            self.W1 -= lr * X.T @ (da * (1-a1**2))
            self.b1 -= lr * (da * (1-a1**2)).sum(0, keepdims=True)

mono = MonoMLP(); mono.train(X, y_oh)

# ═══════════════════════════════════════════════════
# PARADIGM SWARM: one Gaussian per paradigm
# ═══════════════════════════════════════════════════
class GaussianExpert:
    def __init__(self, X_paradigm):
        self.mean = X_paradigm.mean(axis=0)
        self.cov = np.cov(X_paradigm.T)
        self.inv_cov = np.linalg.inv(self.cov)
        self.det = np.linalg.det(self.cov)
        self.dim = X_paradigm.shape[1]
        # Threshold: density at 3-sigma boundary
        d3 = np.exp(-0.5 * 9) / np.sqrt((2*np.pi)**self.dim * self.det)
        self.threshold = d3 * 10  # generous threshold

    def density(self, X):
        """Multivariate Gaussian density"""
        diff = X - self.mean
        mahal = np.sum(diff @ self.inv_cov * diff, axis=1)
        return np.exp(-0.5 * mahal) / np.sqrt((2*np.pi)**self.dim * self.det)

    def is_in_distribution(self, X):
        return self.density(X) > self.threshold

gauss_experts = {}
for i, name in enumerate(names):
    mask = y == i
    gauss_experts[name] = GaussianExpert(X[mask])

# ═══════════════════════════════════════════════════
# TEST ON GRID + SPECIFIC GAP POINTS
# ═══════════════════════════════════════════════════
xx, yy = np.meshgrid(np.linspace(-10, 10, 120), np.linspace(-10, 10, 120))
grid = np.c_[xx.ravel(), yy.ravel()]

# Gap points: deliberately in empty space
gap_test = {
    "A-B midpoint": np.array([0., -5.]),
    "A-C midpoint": np.array([-2.5, 0.5]),
    "B-C midpoint": np.array([2.5, 0.5]),
    "Center (all 3)": np.array([0., 0.]),
    "Far gap NW": np.array([-8., 2.]),
    "Far gap NE": np.array([8., 2.]),
    "Far gap N": np.array([0., 9.]),
    "Far gap S": np.array([0., -9.]),
    "Far gap E": np.array([9., -2.]),
    "Far gap W": np.array([-9., -2.]),
    # In-distribution (should NOT be gap)
    "Labour center": np.array([-5., -5.]),
    "Thermo center": np.array([5., -5.]),
    "Geo center": np.array([0., 6.]),
}

print("="*70)
print("STRONG GAP DETECTION — Gaussian Experts vs Monolithic Softmax")
print("="*70)

print(f"\n{'Point':<22} {'Mono max':>10} {'PS max dens':>12} {'Mono says':>12} {'PS says':>12}")
print("-"*75)

mono_overconf_gap = []
ps_correct_gap = []
total_gap = 0

for name, pt in gap_test.items():
    pt2d = pt.reshape(1, -1)
    mp = mono.forward(pt2d)[0]
    mono_label = list(centers.keys())[mp.argmax()]

    # Gaussian densities
    densities = {n: gauss_experts[n].density(pt2d)[0] for n in names}
    max_dens = max(densities.values())
    in_dist = {n: gauss_experts[n].is_in_distribution(pt2d)[0] for n in names}
    any_in_dist = any(in_dist.values())
    ps_label = "GAP" if not any_in_dist else max(densities, key=densities.get)
    ps_max_norm = max_dens / max(1e-10, sum(densities.values()))  # normalized

    is_gap_point = "gap" in name.lower() or "midpoint" in name.lower() or "center" in name.lower()
    is_in_dist_point = "center" in name.lower() and not ("all" in name.lower() or "gap" in name.lower())

    if is_gap_point and not is_in_dist_point:
        total_gap += 1
        mono_overconf_gap.append(mp.max())
        ps_correct_gap.append(1 if ps_label == "GAP" else 0)

    print(f"{name:<22} {mp.max():>10.3f} {max_dens:>12.6f} {mono_label:<12} {ps_label:<12}")

# ═══════════════════════════════════════════════════
# GRID STATISTICS
# ═══════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"GRID ANALYSIS (120x120 = {len(grid)} points)")
print(f"{'='*70}")

mono_grid = mono.forward(grid).max(axis=1)

# PS: max density across experts
ps_densities = np.column_stack([gauss_experts[n].density(grid) for n in names])
ps_max_dens = ps_densities.max(axis=1)

# Gap = NO expert claims the point
ps_in_dist = np.column_stack([gauss_experts[n].is_in_distribution(grid) for n in names])
ps_gap_mask = ~ps_in_dist.any(axis=1)
gap_pct = 100 * ps_gap_mask.sum() / len(grid)

print(f"\nSpace where PS says GAP:          {gap_pct:.1f}%")
print(f"Mono avg confidence in GAP region: {mono_grid[ps_gap_mask].mean():.3f}")
print(f"PS avg max density in GAP region:  {ps_max_dens[ps_gap_mask].mean():.6f}")
print(f"Mono avg confidence OVERALL:       {mono_grid.mean():.3f}")

# Overconfidence: mono > 0.7 but PS says GAP
overconf = (mono_grid > 0.7) & ps_gap_mask
print(f"Points Mono >0.7 but PS says GAP:  {100*overconf.sum()/len(grid):.1f}%")

# ═══════════════════════════════════════════════════
# VERDICT
# ═══════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"VERDICT")
print(f"{'='*70}")
if mono_overconf_gap:
    print(f"Mono overconfidence on gap points: {np.mean(mono_overconf_gap):.3f} (avg max prob)")
    print(f"PS gap detection rate:              {np.mean(ps_correct_gap):.0%} ({sum(ps_correct_gap)}/{len(ps_correct_gap)})")
print(f"\nGaussian experts NATURALLY leave gaps where density is low.")
print(f"Monolithic softmax is forced to choose — even in empty space.")
print(f"No training on 'gap' examples needed. Structural advantage.")
