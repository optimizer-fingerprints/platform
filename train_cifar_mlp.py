from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from distributed_shampoo import (
    AdamPreconditionerConfig,
    DistributedShampoo,
    EigenConfig,
    PseudoInverseConfig,
    RootInvShampooPreconditionerConfig,
    SingleDeviceDistributedConfig,
    WeightDecayType,
)
from optimizers import MuonWithAuxAdam


class OptimizerBundle:
    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        self.optimizers = optimizers
        self.param_groups = [group for opt in optimizers for group in opt.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for opt in self.optimizers:
            opt.step()


class CifarMLP(nn.Module):
    def __init__(self, hidden_dim: int = 512, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Muon and Shampoo on a small CIFAR-10 MLP.")
    parser.add_argument(
        "--optimizer",
        choices=("muon", "shampoo_default", "shampoo_pinv", "shampoo_pinv_one_sided"),
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--adam-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-test-batches", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_ds = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=train_transform)
    test_ds = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=test_transform)
    pin_memory = str(args.device).startswith("cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def split_params(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    matrix_params: list[nn.Parameter] = []
    aux_params: list[nn.Parameter] = []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2:
            matrix_params.append(p)
        else:
            aux_params.append(p)
    return matrix_params, aux_params


def assert_param_coverage(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    model_params = {p for p in model.parameters() if p.requires_grad}
    opt_params = [p for group in optimizer.param_groups for p in group["params"]]
    if len(opt_params) != len(set(opt_params)):
        raise RuntimeError("A parameter appears in multiple optimizer groups")
    if set(opt_params) != model_params:
        raise RuntimeError("Optimizer parameter groups do not cover trainable model parameters exactly")


def build_shampoo_bundle(
    *,
    args: argparse.Namespace,
    matrix_params: list[nn.Parameter],
    aux_params: list[nn.Parameter],
    preconditioner_config: RootInvShampooPreconditionerConfig,
    epsilon: float,
) -> OptimizerBundle:
    lr = 1e-2 if args.lr is None else args.lr
    beta2 = 0.90
    shampoo = DistributedShampoo(
        matrix_params,
        lr=lr,
        betas=(0.9, beta2),
        epsilon=epsilon,
        weight_decay=args.weight_decay,
        weight_decay_type=WeightDecayType.DECOUPLED,
        max_preconditioner_dim=8192,
        precondition_frequency=1,
        start_preconditioning_step=-1,
        preconditioner_config=preconditioner_config,
        grafting_config=AdamPreconditionerConfig(beta2=beta2, epsilon=1e-15),
        distributed_config=SingleDeviceDistributedConfig(),
    )
    aux_adam = AdamW(aux_params, lr=args.adam_lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    return OptimizerBundle([shampoo, aux_adam])


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer | OptimizerBundle:
    matrix_params, aux_params = split_params(model)
    if args.optimizer == "muon":
        lr = 0.02 if args.lr is None else args.lr
        optimizer = MuonWithAuxAdam(
            [
                {"params": matrix_params, "use_muon": True},
                {"params": aux_params, "use_muon": False},
            ],
            lr=lr,
            weight_decay=args.weight_decay,
            adam_lr=args.adam_lr,
            adam_weight_decay=args.weight_decay,
        )
    elif args.optimizer == "shampoo_default":
        optimizer = build_shampoo_bundle(
            args=args,
            matrix_params=matrix_params,
            aux_params=aux_params,
            preconditioner_config=RootInvShampooPreconditionerConfig(),
            epsilon=1e-12,
        )
    elif args.optimizer in {"shampoo_pinv", "shampoo_pinv_one_sided"}:
        inverse_exponent_override = (
            {2: {0: 0.0, 1: 0.25}} if args.optimizer == "shampoo_pinv_one_sided" else {}
        )
        optimizer = build_shampoo_bundle(
            args=args,
            matrix_params=matrix_params,
            aux_params=aux_params,
            preconditioner_config=RootInvShampooPreconditionerConfig(
                inverse_exponent_override=inverse_exponent_override,
                amortized_computation_config=EigenConfig(
                    rank_deficient_stability_config=PseudoInverseConfig(),
                ),
            ),
            epsilon=0.0,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    assert_param_coverage(model, optimizer)
    return optimizer


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | OptimizerBundle,
    device: torch.device,
    limit_batches: int | None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for batch_idx, (images, targets) in enumerate(tqdm(loader, desc="train", leave=False)):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.numel()
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_seen += targets.numel()
    return total_loss / total_seen, total_correct / total_seen


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    limit_batches: int | None,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for batch_idx, (images, targets) in enumerate(tqdm(loader, desc="test", leave=False)):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, targets)
        total_loss += loss.item() * targets.numel()
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_seen += targets.numel()
    return total_loss / total_seen, total_correct / total_seen


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = build_loaders(args)
    model = CifarMLP(hidden_dim=args.hidden_dim).to(device)
    optimizer = build_optimizer(args, model)
    log_path = args.log_dir / f"cifar_mlp_{args.optimizer}_seed{args.seed}_{int(time.time())}.jsonl"

    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.limit_train_batches,
        )
        test_loss, test_acc = evaluate(model, test_loader, device, args.limit_test_batches)
        elapsed = time.perf_counter() - start
        row = {
            "epoch": epoch,
            "optimizer": args.optimizer,
            "seed": args.seed,
            "lr": args.lr,
            "adam_lr": args.adam_lr,
            "weight_decay": args.weight_decay,
            "hidden_dim": args.hidden_dim,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "elapsed_sec": elapsed,
        }
        with log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row, indent=2))

    print(f"Wrote metrics to {log_path}")


if __name__ == "__main__":
    main()
