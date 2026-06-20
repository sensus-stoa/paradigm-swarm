#!/usr/bin/env python3
"""Generate all figures for Paradigm Swarm paper"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'figure.dpi': 120, 'savefig.dpi': 120, 'savefig.bbox': 'tight'
})

out = '/home/ninjacat/Documents/the_lair/ExoCortex/Journal/figures/'
import os; os.makedirs(out, exist_ok=True)

# ═══════════════════════════════════════════════
# FIGURE 1: Weight Isolation Core Proof
# ═══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 4))
methods = ['Standard (SGD)', 'Paradigm Swarm']
task_a = [51.6, 96.8]
task_b = [93.6, 94.6]
x = np.arange(len(methods))
w = 0.3
ax.bar(x - w/2, task_a, w, label='Task A after B', color='#e74c3c', alpha=0.8)
ax.bar(x + w/2, task_b, w, label='Task B', color='#3498db', alpha=0.8)
ax.set_ylabel('Accuracy (%)')
ax.set_title('Weight Isolation: Core Proof')
ax.set_xticks(x); ax.set_xticklabels(methods)
ax.legend(); ax.set_ylim(0, 110)
for i, (a, b) in enumerate(zip(task_a, task_b)):
    ax.text(i - w/2, a + 1, f'{a}%', ha='center', fontsize=9)
    ax.text(i + w/2, b + 1, f'{b}%', ha='center', fontsize=9)
ax.text(0, 55, 'FORGETTING\n45.2pp', ha='center', fontsize=10, fontweight='bold', color='darkred')
ax.text(1, 96.8, 'FORGETTING\n0.0pp', ha='center', fontsize=10, fontweight='bold', color='darkgreen')
plt.tight_layout(); plt.savefig(out + 'fig1_weight_isolation.png'); plt.close()
print("Fig 1 done")

# ═══════════════════════════════════════════════
# FIGURE 2: Gap Detection
# ═══════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Left: concept
np.random.seed(42)
for i, (c, n) in enumerate([([-5,-5], 'A'), ([5,-5], 'B'), ([0,6], 'C')]):
    pts = np.random.randn(150, 2)*1.2 + np.array(c)
    ax1.scatter(pts[:,0], pts[:,1], s=8, alpha=0.4, label=f'Paradigm {n}')
gap_pts = [[-2.5,0.5], [2.5,0.5], [0,0], [-8,2], [8,2], [0,-9]]
for gp in gap_pts:
    ax1.plot(gp[0], gp[1], 'rx', markersize=10, markeredgewidth=2)
ax1.set_title('Gap Points (red X) between paradigms'); ax1.legend(fontsize=8)

# Right: comparison bar
metrics = ['Gap\nDetection', 'Mono Conf.\nin Gaps', 'Mono >0.7\nbut PS=GAP']
mono_vals = [0, 90.7, 81.5]
ps_vals = [100, 0, 0]
x = np.arange(len(metrics))
ax2.bar(x - 0.15, mono_vals, 0.3, label='Monolithic', color='#e74c3c', alpha=0.8)
ax2.bar(x + 0.15, ps_vals, 0.3, label='Paradigm Swarm', color='#2ecc71', alpha=0.8)
ax2.set_xticks(x); ax2.set_xticklabels(metrics)
ax2.set_title('Gap Detection Performance')
ax2.legend(); ax2.set_ylabel('%')
for i, (m, p) in enumerate(zip(mono_vals, ps_vals)):
    ax2.text(i-0.15, m+1, f'{m}%', ha='center', fontsize=9, fontweight='bold')
    ax2.text(i+0.15, p+1, f'{p}%', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout(); plt.savefig(out + 'fig2_gap_detection.png'); plt.close()
print("Fig 2 done")

# ═══════════════════════════════════════════════
# FIGURE 3: Comprehensive Benchmark
# ═══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
methods = ['Oracle', 'SGD', 'EWC\nλ=5000', 'Replay\n10%', 'PackNet\n50%', 'Paradigm\nSwarm', 'PS wrong\nrouter']
avgs = [0.863, 0.161, 0.182, 0.829, 0.149, 0.861, 0.062]
mins = [0.820, 0.000, 0.000, 0.677, 0.000, 0.820, 0.020]
colors = ['#3498db','#e74c3c','#e67e22','#f39c12','#95a5a6','#2ecc71','#c0392b']
x = np.arange(len(methods))
ax.bar(x, avgs, 0.6, color=colors, alpha=0.85, edgecolor='white')
ax.scatter(x, mins, marker='v', s=80, color='black', zorder=5, label='Min accuracy')
ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel('Accuracy'); ax.set_title('10-Task Sequential Classification (5 seeds)')
ax.legend(); ax.set_ylim(0, 1.05); ax.axhline(y=0.100, color='gray', linestyle='--', alpha=0.5, label='Random')
for i, v in enumerate(avgs):
    ax.text(i, v+0.02, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
    if mins[i] > 0:
        ax.text(i, mins[i]-0.04, f'min:{mins[i]:.3f}', ha='center', fontsize=7, color='black')
plt.tight_layout(); plt.savefig(out + 'fig3_benchmark.png'); plt.close()
print("Fig 3 done")

# ═══════════════════════════════════════════════
# FIGURE 4: Distortion Accumulation (P3 tracking)
# ═══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 4))
steps = ['After P3', 'After P4', 'After P5']
mono = [1.0, 0.24, 0.0]
ps = [1.0, 0.98, 0.96]
ax.plot(steps, mono, 'o-', color='#e74c3c', linewidth=2, markersize=10, label='Monolithic (P3 accuracy)')
ax.plot(steps, ps, 's-', color='#2ecc71', linewidth=2, markersize=10, label='Paradigm Swarm (P3 accuracy)')
ax.fill_between([0,2], [1,1], [0,0], alpha=0.05, color='red')
ax.set_ylabel('P3 Accuracy'); ax.set_title('Distortion Accumulation (I-062)')
ax.legend(); ax.set_ylim(-0.05, 1.1)
for i, (m, p) in enumerate(zip(mono, ps)):
    ax.text(i, m+0.04, f'{m:.2f}', ha='center', fontweight='bold', color='#e74c3c')
    ax.text(i, p-0.08, f'{p:.2f}', ha='center', fontweight='bold', color='#2ecc71')
ax.annotate('P3 FULLY ERASED!', xy=(2, 0.0), xytext=(1.5, 0.2),
            arrowprops=dict(arrowstyle='->', color='red'), fontsize=12, color='red', fontweight='bold')
plt.tight_layout(); plt.savefig(out + 'fig4_distortion.png'); plt.close()
print("Fig 4 done")

# ═══════════════════════════════════════════════
# FIGURE 5: 50-Task Scaling (the killer figure)
# ═══════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

epochs = [20, 40, 80, 160, 320]
sgd_avg = [0.545, 0.529, 0.515, 0.530, 0.527]
ps_avg = [0.509, 0.521, 0.540, 0.705, 0.917]
sgd_min = [0.320, 0.260, 0.330, 0.320, 0.340]
ps_min = [0.370, 0.360, 0.370, 0.430, 0.840]

# Left: avg accuracy
ax1.plot(epochs, sgd_avg, 'o-', color='#e74c3c', linewidth=2, markersize=8, label='SGD (avg)')
ax1.plot(epochs, ps_avg, 's-', color='#2ecc71', linewidth=2, markersize=8, label='Paradigm Swarm (avg)')
ax1.fill_between(epochs, sgd_avg, ps_avg, alpha=0.1, color='green')
ax1.set_xlabel('Epochs per task'); ax1.set_ylabel('Average Accuracy')
ax1.set_title('50-Task Scaling: Average Accuracy'); ax1.legend()
ax1.annotate('SGD: PLATEAU', xy=(320, 0.527), xytext=(200, 0.45),
            arrowprops=dict(arrowstyle='->'), fontsize=10, color='#e74c3c', fontweight='bold')
ax1.annotate('PS: NO CEILING', xy=(320, 0.917), xytext=(200, 0.85),
            arrowprops=dict(arrowstyle='->'), fontsize=10, color='#2ecc71', fontweight='bold')

# Right: T0 forgetting
sgd_t0_drop = [0.06, -0.02, 0.13, 0.50, 0.57]
ax2.bar(np.arange(5)-0.15, sgd_t0_drop, 0.3, color='#e74c3c', alpha=0.8, label='SGD T0 forgetting')
ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax2.set_xticks(np.arange(5)); ax2.set_xticklabels(epochs)
ax2.set_xlabel('Epochs per task'); ax2.set_ylabel('Task 0 Accuracy Drop (pp)')
ax2.set_title('More Training = MORE Forgetting (SGD)')
for i, v in enumerate(sgd_t0_drop):
    color = 'darkred' if v > 0.3 else 'black'
    ax2.text(i-0.15, v+0.03 if v>0 else v-0.06, f'{v:+.2f}', ha='center', fontweight='bold', color=color, fontsize=10)
ax2.text(3.5, 0.57, '57pp\nERASED', ha='center', fontsize=12, color='darkred', fontweight='bold')

plt.tight_layout(); plt.savefig(out + 'fig5_scaling.png'); plt.close()
print("Fig 5 done")

# ═══════════════════════════════════════════════
# FIGURE 6: Trade-off summary
# ═══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4))
benchmarks = ['2D Gaussians\n(10 tasks)', 'Split-Pattern\n(10 tasks)', 'Split-MNIST\n(2 tasks)', 'Split-CIFAR\n(2 tasks)', 'CIFAR 5x2\n(5 tasks)', '50 Synthetic\n(50 tasks)']
sgd_vals = [0.161, 0.333, 0.762, 0.453, 0.536, 0.527]
ps_vals = [0.861, 1.000, 0.779, 0.429, 0.675, 0.917]
gap = [p-s for p,s in zip(ps_vals, sgd_vals)]

colors = ['#2ecc71' if g > 0 else '#e74c3c' for g in gap]
x = np.arange(len(benchmarks))
bars = ax.bar(x, gap, 0.6, color=colors, alpha=0.85, edgecolor='white')
ax.axhline(y=0, color='black', linewidth=1)
ax.set_xticks(x); ax.set_xticklabels(benchmarks, fontsize=9)
ax.set_ylabel('PS Advantage over SGD (pp)'); ax.set_title('When Paradigm Swarm Wins (and When It Doesn\'t)')
for i, (g, s, p) in enumerate(zip(gap, sgd_vals, ps_vals)):
    y = g + (0.03 if g > 0 else -0.06)
    ax.text(i, y, f'{g:+.3f}\n(SGD:{s:.2f} PS:{p:.2f})', ha='center', fontsize=7, fontweight='bold')
ax.text(4.7, 0.5, 'TRANSFER >\nINTERFERENCE', ha='center', fontsize=9, color='darkred')
ax.text(0.5, 0.65, 'INTERFERENCE >\nTRANSFER', ha='center', fontsize=9, color='darkgreen')
plt.tight_layout(); plt.savefig(out + 'fig6_tradeoff.png'); plt.close()
print("Fig 6 done")

print(f"\nAll 6 figures saved to {out}")
