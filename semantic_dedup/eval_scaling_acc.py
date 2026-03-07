#!/usr/bin/env python3
"""
Evaluate SemDeDup scaling: train ResNet18 on deduplicated CIFAR-10 at eps=0.05
for each scaling ratio (0.7, 0.5, 0.3) and both baseline/optimized dedup outputs.

Reuses existing dedup outputs in output_scaling_v2/ by reading per-cluster
dataframes and filtering by eps=0.05 column.
"""

import os
import json
import time
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models


# ============================================================================
# Configuration
# ============================================================================
BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
DATA_ROOT = "./data"
EPS = 0.05

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================================
# Data transforms
# ============================================================================
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def create_model():
    """Create ResNet18 for CIFAR-10 (modified for 32x32 images)."""
    model = models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model.to(device)


def train_epoch(model, trainloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return running_loss / len(trainloader), 100. * correct / total


def evaluate(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total


def train_and_evaluate(trainset, testloader, name, seed=42):
    """Full training loop. Returns best test accuracy."""
    torch.manual_seed(seed)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_acc = 0
    t0 = time.time()
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer)
        test_acc = evaluate(model, testloader)
        scheduler.step()
        if test_acc > best_acc:
            best_acc = test_acc
        if (epoch + 1) % 5 == 0:
            print(f"  [{name}] Epoch {epoch+1}/{NUM_EPOCHS}: train_acc={train_acc:.2f}%, test_acc={test_acc:.2f}%")
    elapsed = time.time() - t0
    print(f"  [{name}] Best: {best_acc:.2f}%, Time: {elapsed:.1f}s")
    return best_acc


def extract_kept_indices(dedup_dir, num_clusters, eps=0.05):
    """Extract kept indices at a given eps from existing dedup dataframes."""
    kept = []
    dataframes_dir = os.path.join(dedup_dir, 'dataframes')
    sorted_clusters_dir = os.path.join(dedup_dir, 'sorted_clusters')
    
    for cid in range(num_clusters):
        cluster_npy = os.path.join(sorted_clusters_dir, f'cluster_{cid}.npy')
        df_file = os.path.join(dataframes_dir, f'cluster_{cid}.pkl')

        if not os.path.exists(cluster_npy) or not os.path.exists(df_file):
            continue

        cluster_data = np.load(cluster_npy, allow_pickle=True)
        if len(cluster_data) == 0:
            continue

        orig_indices = cluster_data[:, 1].astype(int)

        with open(df_file, 'rb') as f:
            df = pickle.load(f)

        eps_col = f'eps={eps}'
        if eps_col not in df.columns:
            print(f"  Warning: {eps_col} not in cluster {cid}, available: {df.columns.tolist()}")
            continue

        for i, orig_idx in enumerate(orig_indices):
            if not df[eps_col].iloc[i]:  # False = not duplicate = keep
                kept.append(int(orig_idx))

    return sorted(kept)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dedup-dir', default='./output_scaling_v2',
                        help='Directory with scaling dedup outputs')
    parser.add_argument('--ratios', nargs='+', type=float, default=[0.7, 0.5, 0.3])
    parser.add_argument('--num-clusters', type=int, default=10)
    parser.add_argument('--eps', type=float, default=0.05)
    parser.add_argument('--output', type=str, default='./output_scaling_v2/scaling_acc_results.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Load CIFAR-10
    full_trainset = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    full_labels = np.array(full_trainset.targets)

    results = {}

    for ratio in args.ratios:
        print(f"\n{'='*60}")
        print(f"RATIO={ratio}")
        print(f"{'='*60}")

        # Recreate the subset indices (same seed as scaling experiment)
        np.random.seed(args.seed)
        subset_indices = []
        for cls in range(10):
            cls_indices = np.where(full_labels == cls)[0]
            n_per_class = int(len(cls_indices) * ratio)
            chosen = np.random.choice(cls_indices, n_per_class, replace=False)
            subset_indices.extend(chosen.tolist())
        subset_indices = sorted(subset_indices)

        ratio_results = {'ratio': ratio, 'n_samples': len(subset_indices)}

        # Baseline dedup kept indices at eps=0.05
        baseline_dir = os.path.join(args.dedup_dir, f'ratio_{ratio}_baseline')
        baseline_kept = extract_kept_indices(baseline_dir, args.num_clusters, args.eps)
        # Map subset-local indices to full-dataset indices
        baseline_kept_global = [subset_indices[i] for i in baseline_kept if i < len(subset_indices)]
        
        print(f"  Baseline kept: {len(baseline_kept_global)} / {len(subset_indices)} ({100*len(baseline_kept_global)/len(subset_indices):.1f}%)")

        # Train on baseline dedup
        baseline_trainset = Subset(full_trainset, baseline_kept_global)
        baseline_acc = train_and_evaluate(baseline_trainset, testloader, f"baseline_r{ratio}", args.seed)
        ratio_results['baseline_acc'] = baseline_acc
        ratio_results['baseline_kept'] = len(baseline_kept_global)

        # Optimized dedup kept indices at eps=0.05 (use run1)
        opt_dir = os.path.join(args.dedup_dir, f'ratio_{ratio}_opt_run1')
        opt_kept = extract_kept_indices(opt_dir, args.num_clusters, args.eps)
        opt_kept_global = [subset_indices[i] for i in opt_kept if i < len(subset_indices)]

        print(f"  Optimized kept: {len(opt_kept_global)} / {len(subset_indices)} ({100*len(opt_kept_global)/len(subset_indices):.1f}%)")

        opt_trainset = Subset(full_trainset, opt_kept_global)
        opt_acc = train_and_evaluate(opt_trainset, testloader, f"optimized_r{ratio}", args.seed)
        ratio_results['optimized_acc'] = opt_acc
        ratio_results['optimized_kept'] = len(opt_kept_global)

        results[str(ratio)] = ratio_results

    # Summary
    print(f"\n{'='*60}")
    print(f"ACC@eps={args.eps} SUMMARY")
    print(f"{'='*60}")
    for r, d in sorted(results.items(), key=lambda x: -float(x[0])):
        print(f"  ratio={r}: baseline_acc={d['baseline_acc']:.2f}% ({d['baseline_kept']} kept), "
              f"opt_acc={d['optimized_acc']:.2f}% ({d['optimized_kept']} kept)")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
