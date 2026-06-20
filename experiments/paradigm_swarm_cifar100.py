#!/usr/bin/env python3
"""CIFAR-100 ResNet-18: Paradigm Swarm vs SGD — Sequential Learning

Trivedi & Melwani (2026) scale. CIFAR-100, 2 tasks × 50 classes.
ResNet-18 backbone. 20 epochs/task. CPU training.
PS: 2 isolated ResNet-18 experts. SGD: shared backbone, sequential.

Key question: does forward transfer on CIFAR compensate for interference
at this scale (100 classes, ResNet), or does SGD degrade?
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle, urllib.request, tarfile, io, os, sys

torch.set_num_threads(8)
DEVICE = torch.device('cpu')
BATCH = 64
EPOCHS = 5
LR = 0.01

# ═══════════════════════════════════════
# DOWNLOAD CIFAR-100
# ═══════════════════════════════════════
def load_cifar100():
    cache_path = '/tmp/cifar100.npz'
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        X_tr, y_tr = d['X_tr'], d['y_tr']
        X_te, y_te = d['X_te'], d['y_te']
        # Reshape flat 3072 → (N, 3, 32, 32) if needed
        if X_tr.ndim == 2:
            X_tr = X_tr.reshape(-1, 3, 32, 32)
            X_te = X_te.reshape(-1, 3, 32, 32)
        return X_tr, y_tr, X_te, y_te
    
    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    print('Downloading CIFAR-100 (~169MB)...')
    resp = urllib.request.urlopen(url)
    data = resp.read()
    tar = tarfile.open(fileobj=io.BytesIO(data))
    
    f = tar.extractfile('cifar-100-python/train')
    batch = pickle.load(f, encoding='bytes')
    X_tr = batch[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y_tr = np.array(batch[b'fine_labels']).astype(np.int64)
    
    f = tar.extractfile('cifar-100-python/test')
    batch = pickle.load(f, encoding='bytes')
    X_te = batch[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y_te = np.array(batch[b'fine_labels']).astype(np.int64)
    
    np.savez(cache_path, X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te)
    print(f'Cached: {cache_path}')
    return X_tr, y_tr, X_te, y_te

X_tr, y_tr, X_te, y_te = load_cifar100()

# Split into 2 tasks: classes 0-49, 50-99
TASK_SPLIT = 50
# Task 0: classes 0-49, Task 1: classes 50-99

def filter_task(X, y, task_id):
    lo, hi = task_id * TASK_SPLIT, (task_id + 1) * TASK_SPLIT
    mask = (y >= lo) & (y < hi)
    return X[mask], y[mask] - lo  # remap to 0..49

# ═══════════════════════════════════════
# RESNET-18 (from torchvision, handwritten for no-dependency)
# ═══════════════════════════════════════
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=50):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out); out = self.layer2(out)
        out = self.layer3(out); out = self.layer4(out)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        return self.linear(out)

def ResNet18(num_classes=50):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

# ═══════════════════════════════════════
# TRAIN ONE EPOCH
# ═══════════════════════════════════════
def train_epoch(model, X, y, opt, criterion):
    model.train()
    n = len(X)
    perm = torch.randperm(n)
    losses = []
    for i in range(0, n, BATCH):
        idx = perm[i:i+BATCH]
        xb = torch.from_numpy(X[idx]).to(DEVICE)
        yb = torch.from_numpy(y[idx]).to(DEVICE)
        opt.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return np.mean(losses)

def evaluate(model, X, y):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(X), BATCH * 2):
            xb = torch.from_numpy(X[i:i+BATCH*2]).to(DEVICE)
            yb = torch.from_numpy(y[i:i+BATCH*2]).to(DEVICE)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
    return correct / len(X)

# ═══════════════════════════════════════
# EXPERIMENT
# ═══════════════════════════════════════
print("=" * 65)
print("CIFAR-100 ResNet-18: Paradigm Swarm vs SGD")
print(f"2 tasks × 50 classes, {EPOCHS} epochs/task, CPU")
print("=" * 65)

# ═══ SGD: shared backbone, 2 output heads ═══
print("\n─── SGD (shared backbone, sequential) ───")
sgd_model = ResNet18(num_classes=50)

# Task 0
X0_tr, y0_tr = filter_task(X_tr, y_tr, 0)
X0_te, y0_te = filter_task(X_te, y_te, 0)
opt0 = optim.SGD(sgd_model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

print(f"  Task 0: {len(X0_tr)} train, {len(X0_te)} test")
for ep in range(EPOCHS):
    loss = train_epoch(sgd_model, X0_tr, y0_tr, opt0, criterion)
    if ep % 10 == 9 or ep == 0:
        acc = evaluate(sgd_model, X0_te, y0_te)
        print(f"    ep {ep+1:3d}: loss={loss:.3f} acc={acc:.3f}")

sgd_t0_acc = evaluate(sgd_model, X0_te, y0_te)
print(f"  Task 0 final: {sgd_t0_acc:.3f}")

# Task 1 — overwrites backbone!
X1_tr, y1_tr = filter_task(X_tr, y_tr, 1)
X1_te, y1_te = filter_task(X_te, y_te, 1)
opt1 = optim.SGD(sgd_model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

print(f"\n  Task 1: {len(X1_tr)} train, {len(X1_te)} test")
for ep in range(EPOCHS):
    loss = train_epoch(sgd_model, X1_tr, y1_tr, opt1, criterion)
    if ep % 10 == 9 or ep == 0:
        acc = evaluate(sgd_model, X1_te, y1_te)
        print(f"    ep {ep+1:3d}: loss={loss:.3f} acc={acc:.3f}")

sgd_t1_acc = evaluate(sgd_model, X1_te, y1_te)
sgd_t0_after = evaluate(sgd_model, X0_te, y0_te)
sgd_forget = sgd_t0_acc - sgd_t0_after
print(f"\n  SGD final: T0={sgd_t0_after:.3f} (was {sgd_t0_acc:.3f}, forgot {sgd_forget:.3f})  T1={sgd_t1_acc:.3f}")

# ═══ Paradigm Swarm: 2 isolated ResNet-18 experts ═══
print("\n─── Paradigm Swarm (2 isolated ResNet-18 experts) ───")

# Expert 0
ps0 = ResNet18(num_classes=50)
opt_ps0 = optim.SGD(ps0.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
for ep in range(EPOCHS):
    loss = train_epoch(ps0, X0_tr, y0_tr, opt_ps0, criterion)
ps_t0 = evaluate(ps0, X0_te, y0_te)
print(f"  Expert 0: {ps_t0:.3f}")

# Expert 1 — isolated, no interference possible
ps1 = ResNet18(num_classes=50)
opt_ps1 = optim.SGD(ps1.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
for ep in range(EPOCHS):
    loss = train_epoch(ps1, X1_tr, y1_tr, opt_ps1, criterion)
ps_t1 = evaluate(ps1, X1_te, y1_te)

# Expert 0 accuracy unchanged
ps_t0_after = evaluate(ps0, X0_te, y0_te)
print(f"  Expert 1: {ps_t1:.3f}")
print(f"  Expert 0 after T1: {ps_t0_after:.3f} (unchanged: {ps_t0_after == ps_t0})")

# ═══════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════
print(f"\n{'=' * 65}")
print("SUMMARY")
print(f"{'=' * 65}")
print(f"{'':<20} {'SGD':>10} {'PS':>10}")
print("-" * 42)
print(f"{'Task 0 acc':<20} {sgd_t0_after:>10.3f} {ps_t0_after:>10.3f}")
print(f"{'Task 1 acc':<20} {sgd_t1_acc:>10.3f} {ps_t1:>10.3f}")
print(f"{'Avg':<20} {np.mean([sgd_t0_after,sgd_t1_acc]):>10.3f} {np.mean([ps_t0_after,ps_t1]):>10.3f}")
print(f"{'Task 0 forgetting':<20} {sgd_forget:>+10.3f} {'0.000':>10}")
