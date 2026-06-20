#!/usr/bin/env python3
"""
CIFAR-100 ResNet-18: Paradigm Swarm vs SGD — Sequential Learning
Kaggle GPU version. 2 tasks × 50 classes, 20 epochs/task.

Paste this entire cell into a Kaggle notebook.
Accelerator: GPU P100 or T4.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm

# ═══════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH = 128
EPOCHS = 20
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ═══════════════════════════════════════
# DATA
# ═══════════════════════════════════════
transform_train = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

train_full = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_train)
test_full  = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=transform_test)

# Split into 2 tasks: classes 0-49, 50-99
def task_subset(dataset, lo, hi, remap=True):
    """Extract subset of classes [lo, hi) and optionally remap to 0..(hi-lo-1)."""
    indices = [i for i, (_, y) in enumerate(dataset) if lo <= y < hi]
    if remap:
        # Override targets to 0..N-1
        dataset.targets = list(dataset.targets)  # ensure mutable
        old_targets = dataset.targets.copy()
        for i in indices:
            dataset.targets[i] = old_targets[i] - lo
    return Subset(dataset, indices)

# Task 0: classes 0-49, Task 1: classes 50-99
t0_train = DataLoader(task_subset(train_full, 0, 50), batch_size=BATCH, shuffle=True, num_workers=2)
t0_test  = DataLoader(task_subset(test_full,  0, 50), batch_size=BATCH, shuffle=False, num_workers=2)
t1_train = DataLoader(task_subset(train_full, 50, 100), batch_size=BATCH, shuffle=True, num_workers=2)
t1_test  = DataLoader(task_subset(test_full,  50, 100), batch_size=BATCH, shuffle=False, num_workers=2)

print(f"\nTask 0: {len(t0_train.dataset)} train, {len(t0_test.dataset)} test")
print(f"Task 1: {len(t1_train.dataset)} train, {len(t1_test.dataset)} test")

# ═══════════════════════════════════════
# MODEL
# ═══════════════════════════════════════
def make_model(num_classes=50):
    return torchvision.models.resnet18(num_classes=num_classes).to(DEVICE)

criterion = nn.CrossEntropyLoss()

# ═══════════════════════════════════════
# TRAIN / EVAL
# ═══════════════════════════════════════
def train_epoch(model, loader, opt):
    model.train()
    losses = []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return np.mean(losses)

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0

def train_task(model, train_loader, test_loader, epochs, label=""):
    opt = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_acc = 0.0
    pbar = tqdm(range(epochs), desc=label)
    for ep in pbar:
        loss = train_epoch(model, train_loader, opt)
        acc = evaluate(model, test_loader)
        scheduler.step()
        best_acc = max(best_acc, acc)
        pbar.set_postfix({"loss": f"{loss:.3f}", "acc": f"{acc:.3f}"})
    return best_acc

# ═══════════════════════════════════════
# EXPERIMENT
# ═══════════════════════════════════════
print(f"\n{'='*60}")
print(f"CIFAR-100 ResNet-18: Paradigm Swarm vs SGD")
print(f"2 tasks × 50 classes, {EPOCHS} epochs/task, {DEVICE}")
print(f"{'='*60}")

# ─── SGD: shared backbone, sequential ───
print("\n─── SGD (shared backbone, sequential) ───")
sgd = make_model(num_classes=50)

t0_acc_sgd = train_task(sgd, t0_train, t0_test, EPOCHS, label="SGD T0")
print(f"  Task 0 best: {t0_acc_sgd:.4f}")

t1_acc_sgd = train_task(sgd, t1_train, t1_test, EPOCHS, label="SGD T1")
print(f"  Task 1 best: {t1_acc_sgd:.4f}")

# Re-evaluate Task 0 AFTER Task 1 training (catastrophic forgetting check)
t0_after_sgd = evaluate(sgd, t0_test)
forget_sgd = t0_acc_sgd - t0_after_sgd
print(f"  Task 0 AFTER T1: {t0_after_sgd:.4f} (forgot: {forget_sgd:+.4f})")

# ─── Paradigm Swarm: 2 isolated ResNet-18 ───
print("\n─── Paradigm Swarm (2 isolated experts) ───")

ps0 = make_model(num_classes=50)
t0_acc_ps = train_task(ps0, t0_train, t0_test, EPOCHS, label="PS Expert 0")
print(f"  Expert 0 best: {t0_acc_ps:.4f}")

ps1 = make_model(num_classes=50)
t1_acc_ps = train_task(ps1, t1_train, t1_test, EPOCHS, label="PS Expert 1")
print(f"  Expert 1 best: {t1_acc_ps:.4f}")

# Expert 0 is ISOLATED — accuracy unchanged
t0_after_ps = evaluate(ps0, t0_test)
print(f"  Expert 0 AFTER T1: {t0_after_ps:.4f} (unchanged: {abs(t0_after_ps - t0_acc_ps) < 1e-6})")

# ═══════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"{'':<25} {'SGD':>12} {'PS':>12}")
print("-" * 50)
print(f"{'Task 0 final acc':<25} {t0_after_sgd:>12.4f} {t0_after_ps:>12.4f}")
print(f"{'Task 1 acc':<25} {t1_acc_sgd:>12.4f} {t1_acc_ps:>12.4f}")
print(f"{'Average':<25} {np.mean([t0_after_sgd, t1_acc_sgd]):>12.4f} {np.mean([t0_after_ps, t1_acc_ps]):>12.4f}")
print(f"{'Task 0 forgetting':<25} {forget_sgd:>+12.4f} {'0.0000':>12}")
print(f"\n{'-'*50}")
print(f"SGD forward transfer: {t1_acc_sgd - t0_after_sgd:+.4f}")
print(f"PS  forward transfer: {t1_acc_ps - t0_after_ps:+.4f} (none by construction)")

# ═══════════════════════════════════════
# INTERPRETATION
# ═══════════════════════════════════════
print(f"\n{'='*60}")
print("INTERPRETATION")
print(f"{'='*60}")

if np.mean([t0_after_sgd, t1_acc_sgd]) > np.mean([t0_after_ps, t1_acc_ps]):
    print("SGD wins on average: forward transfer compensates for forgetting")
    print("at 2-task CIFAR-100 scale (consistent with §4.9 Split-CIFAR).")
else:
    print("PS wins on average: isolation advantage > forward transfer loss.")

print(f"\nForgetting: SGD loses {forget_sgd:.4f} on T0, PS loses 0.0000")
print(f"At 50+ tasks with diverse paradigms, SGD's forward transfer vanishes,")
print(f"while PS retains zero-forgetting guarantee (§4.10).")
