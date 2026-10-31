"""Decode block-19 pooled features into 16 RGB frames at 64 x 64.

Run on a compute node: python train_decoder.py
The decoder uses its own fixed 80/20 split, independent of probe splits A/B/C/D.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils import load_data, load_features, load_video


ROOT = Path(__file__).resolve().parents[1]
FEATURES = Path('/scratch/users/hmolin/jepa-interp/pooled_features')
BLOCK = 19
SEED = 42


class Decoder(nn.Module):
    """Raw pooled features [B, 1024] -> RGB clips [B, 16, 3, 64, 64]."""

    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.zeros(1024))
        self.register_buffer('scale', torch.ones(1024))
        self.network = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Linear(256, 64 * 4 * 4), nn.ReLU(),
            nn.Unflatten(1, (64, 4, 4)),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16 * 3, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, features):
        normalized = (features - self.mean) / self.scale
        return self.network(normalized).reshape(-1, 16, 3, 64, 64)


def load_examples():
    """Decode videos once into ~900 MB of uint8 RAM; do not change source data."""
    data = load_data(ROOT)
    count = sum(len(ds['ids']) for ds in data.values())
    features = np.empty((count, 1024), dtype=np.float32)
    clips = torch.empty((count, 16, 3, 64, 64), dtype=torch.uint8)
    identities, train, test = [], [], []
    rng = np.random.default_rng(SEED)
    offset = 0
    for task, ds in data.items():
        n = len(ds['ids'])
        features[offset:offset + n] = load_features(
            FEATURES / f'{task}.npz', ds['ids'])[:, BLOCK - 1]
        order = rng.permutation(n) + offset
        cut = int(.8 * n)
        train.extend(order[:cut].tolist())
        test.extend(order[cut:].tolist())
        for j, (video, video_id) in enumerate(zip(ds['videos'], ds['ids'])):
            rgb = torch.from_numpy(load_video(video)).permute(0, 3, 1, 2).float()
            small = F.interpolate(rgb, size=(64, 64), mode='area')
            clips[offset + j] = small.round().to(torch.uint8)
            identities.append({'task': task, 'id': int(video_id)})
            if (j + 1) % 250 == 0:
                print(f'{task}: decoded {j + 1}/{n}', flush=True)
        offset += n
    return torch.from_numpy(features), clips, identities, train, test


def reconstruction_loss(prediction, target):
    # Background is ~0.12; the bright disk exceeds 0.2. Weight is target-only.
    foreground = target.amax(dim=2, keepdim=True) > .2
    weights = 1 + 19 * foreground.to(target.dtype)
    squared_error = (prediction - target).square()
    return (squared_error * weights).mean(), squared_error.mean()


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    totals = np.zeros(2)
    count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for features, pixels in loader:
            features = features.to(device)
            target = pixels.to(device, dtype=torch.float32) / 255
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            weighted, mse = reconstruction_loss(model(features), target)
            if optimizer is not None:
                weighted.backward()
                optimizer.step()
            totals += np.array([weighted.item(), mse.item()]) * len(features)
            count += len(features)
    return dict(zip(('weighted_mse', 'mse'), (totals / count).tolist()))


def save_preview(model, features, clips, identities, indices, device, path):
    model.eval()
    with torch.no_grad():
        predictions = model(features[indices].to(device)).cpu()
    fig, axes = plt.subplots(2 * len(indices), 8, figsize=(12, 3 * len(indices)))
    for row, index in enumerate(indices):
        for col, frame in enumerate(range(0, 16, 2)):
            axes[2 * row, col].imshow(clips[index, frame].permute(1, 2, 0))
            axes[2 * row + 1, col].imshow(predictions[row, frame].permute(1, 2, 0))
            axes[2 * row, col].set_title(f'Frame {frame + 1}', fontsize=8)
        identity = identities[index]
        axes[2 * row, 0].set_ylabel(f'{identity["task"]} {identity["id"]}\nOriginal')
        axes[2 * row + 1, 0].set_ylabel('Decoded')
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/decoder')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    # A fresh directory prevents accidental overwriting of a previous experiment.
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}; loading videos', flush=True)
    features, clips, identities, train, test = load_examples()
    protocol = {
        'block': BLOCK, 'seed': SEED, 'epochs': args.epochs,
        'resolution': 64, 'frames': 16, 'batch_size': 32,
        'learning_rate': .001, 'weight_decay': .0001,
        'foreground_threshold': .2, 'foreground_weight': 20,
        'feature_directory': str(FEATURES),
        'train': [identities[i] for i in train],
        'test': [identities[i] for i in test],
    }
    (args.output / 'split.json').write_text(json.dumps(protocol, indent=2))
    dataset = TensorDataset(features, clips)
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train),
                              batch_size=32, shuffle=True)
    test_loader = DataLoader(torch.utils.data.Subset(dataset, test), batch_size=32)
    model = Decoder().to(device)
    model.mean.copy_(features[train].mean(0).to(device))
    model.scale.copy_(features[train].std(0).clamp_min(1e-6).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    print(f'{len(train)} train / {len(test)} test; '
          f'{sum(p.numel() for p in model.parameters()):,} parameters', flush=True)
    # Fixed preview examples: one from each dataset, chosen before fitting.
    preview = [next(i for i in test if identities[i]['task'] == task)
               for task in ('direction', 'speed', 'acceleration')]
    history = []
    for epoch in range(1, args.epochs + 1):
        training = run_epoch(model, train_loader, device, optimizer)
        testing = run_epoch(model, test_loader, device)
        history.append({'epoch': epoch, 'train': training, 'test': testing})
        print(f'Epoch {epoch:03d}: weighted MSE '
              f'train={training["weighted_mse"]:.6f} test={testing["weighted_mse"]:.6f}; '
              f'test pixel MSE={testing["mse"]:.6f}', flush=True)
        (args.output / 'history.json').write_text(json.dumps(history, indent=2))
        # Always save the latest epoch; test loss does not select a checkpoint.
        torch.save({'model': model.state_dict(), 'block': BLOCK, 'epoch': epoch},
                   args.output / 'decoder.pt')
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            save_preview(model, features, clips, identities, preview, device,
                         args.output / 'reconstructions.png')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for ax, metric in zip(axes, ('weighted_mse', 'mse')):
        for split in ('train', 'test'):
            ax.plot([h['epoch'] for h in history],
                    [h[split][metric] for h in history], label=split)
        ax.set(xlabel='Epoch', ylabel=metric)
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / 'loss.png', dpi=150)
    plt.close(fig)
    print(f'Saved decoder and reconstructions to {args.output}', flush=True)


if __name__ == '__main__':
    main()
