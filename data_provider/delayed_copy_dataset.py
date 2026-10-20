"""Reproducible multivariate delayed-copying dataset for sequence classification.

The memory symbol is shown only at t=0. The query flag appears only at t=T-1.
All intermediate steps contain no target information or labels. The classification
label is available to the training loop only for the endpoint loss.

Example:
    python delayed_copy_dataset.py --seq-len 200 --feature-dim 16 --num-classes 8
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class DelayedCopyDataset(Dataset):
    """Return (sequence [T,F], label [1], padding_mask [T]).

    The memory symbol is shown only at t=0. The query flag appears only at t=T-1.
    All intermediate steps contain no target information or labels. The classification
    label is available to the training loop only for the endpoint loss.

    Feature layout:
      [0:num_classes]   one-hot memory code at t=0
      [-1]              query flag at t=T-1
      remaining channels zero or random distractors during intervening steps

    For this simple one-hot construction, F must be >= num_classes + 1.
    """

    def __init__(
        self,
        size: int = 50_000,
        seq_len: int = 200,
        feature_dim: int = 16,
        num_classes: int = 8,
        noise_std: float = 0.0,
        seed: int = 2026,
    ) -> None:
        if size <= 0 or seq_len < 3 or num_classes < 2:
            raise ValueError('Require size > 0, seq_len >= 3, num_classes >= 2')
        if feature_dim < num_classes + 1:
            raise ValueError('feature_dim must be at least num_classes + 1 (classes + query channel)')
        if noise_std < 0:
            raise ValueError('noise_std must be >= 0')
        self.size, self.seq_len, self.feature_dim = size, seq_len, feature_dim
        self.num_classes, self.noise_std, self.seed = num_classes, noise_std, seed
        rng = np.random.default_rng(seed)
        self.labels = rng.integers(0, num_classes, size=size, dtype=np.int64)
        # Dataset indices deterministically generate noise, independent of DataLoader workers/order.

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        label = int(self.labels[index])
        x = np.zeros((self.seq_len, self.feature_dim), dtype=np.float32)
        # Add noise to non-symbol channels only so that the answer is never repeated.
        spare_start = self.num_classes
        spare_end = self.feature_dim - 1
        if self.noise_std > 0 and spare_end > spare_start:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index), 12345]))
            x[1:-1, spare_start:spare_end] = rng.normal(
                0.0, self.noise_std, size=(self.seq_len - 2, spare_end - spare_start)
            ).astype(np.float32)
        x[0, label] = 1.0
        x[-1, -1] = 1.0
        mask = np.ones(self.seq_len, dtype=np.float32)
        return torch.from_numpy(x), torch.tensor([label], dtype=torch.long), torch.from_numpy(mask)


class SavedDelayedCopyDataset(DelayedCopyDataset):
    """Load a generated train.pt / val.pt / test.pt file."""

    def __init__(self, pt_path):
        data = torch.load(pt_path, map_location="cpu", weights_only=True)

        labels = data["labels"].long().view(-1)
        config = data["config"]

        super().__init__(
            size=len(labels),
            seq_len=config["seq_len"],
            feature_dim=config["feature_dim"],
            num_classes=config["num_classes"],
            noise_std=config["noise_std"],
            seed=config["seed"],
        )

        # Use saved labels, not newly sampled ones.
        self.labels = labels.numpy().copy()


def main():
    p = argparse.ArgumentParser(description='Generate multivariate delayed-copy data (endpoint classification).')
    p.add_argument('--seq-len', type=int, default=100)
    p.add_argument('--feature-dim', type=int, default=16)
    p.add_argument('--num-classes', type=int, default=8)
    p.add_argument('--train-size', type=int, default=50_000)
    p.add_argument('--val-size', type=int, default=10_000)
    p.add_argument('--test-size', type=int, default=10_000)
    p.add_argument('--noise-std', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=2026)
    # p.add_argument('--output-dir', type=Path, default=Path('delayed_copy_data'))
    p.add_argument('--output-dir', type=Path,
               default=None, help='Directory to save generated dataset files (train.pt, val.pt, test.pt).')
    p.add_argument('--format', choices=['pt', 'csv', 'txt'], default='pt', help='File export format')
    args = p.parse_args()
    
    if args.output_dir is None:
        args.output_dir = (
            Path(__file__).resolve().parent.parent
            / 'dataset'
            / f'delayed_copy_seq_{args.seq_len}_noise_{args.noise_std}_class_{args.num_classes}'
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, count, split_seed in [
        ('train', args.train_size, args.seed),
        ('val', args.val_size, args.seed + 1),
        ('test', args.test_size, args.seed + 2),
    ]:
        dataset = DelayedCopyDataset(
            size=count, seq_len=args.seq_len, feature_dim=args.feature_dim,
            num_classes=args.num_classes, noise_std=args.noise_std, seed=split_seed,
        )
        out = args.output_dir / f'{split}.{args.format}'
        if args.format == 'pt':
            # Save index labels and config only: reconstruct examples via Dataset.
            torch.save({
                'labels': torch.from_numpy(dataset.labels.copy()),
                'config': dict(seq_len=args.seq_len, feature_dim=args.feature_dim,
                               num_classes=args.num_classes, noise_std=args.noise_std, seed=split_seed),
            }, out)
        else:
            # readable exports: one row per sample, flattened in time-major order.
            # The first two columns are sample_id, label; reshape features to [T,F] when reading.
            sep = ',' if args.format == 'csv' else '\t'
            with out.open('w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f, delimiter=sep)
                writer.writerow(['sample_id', 'label'] + [f't{t}_f{j}' for t in range(args.seq_len)
                                                        for j in range(args.feature_dim)])
                for idx in range(count):
                    x, y, _ = dataset[idx]
                    writer.writerow([idx, int(y.item())] + x.reshape(-1).tolist())
        print(f'{split}: {count} examples -> {out}')
    example = DelayedCopyDataset(1, args.seq_len, args.feature_dim, args.num_classes, args.noise_std, args.seed)
    x, y, mask = example[0]
    print('Example shapes:', tuple(x.shape), tuple(y.shape), tuple(mask.shape))
    print('Label:', int(y.item()), 'memory:', x[0].tolist(), 'query flag:', x[-1, -1].item())
    print('Note: .pt stores labels/config only; CSV/TXT exports contain the complete flattened inputs.')


if __name__ == '__main__':
    main()
