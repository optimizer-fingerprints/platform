# fingerprinting

Small optimizer experiments.

## Muon vs Shampoo on CIFAR-10

Install dependencies:

```bash
uv sync
```

Smoke test Muon:

```bash
uv run python train_cifar_mlp.py --optimizer muon --epochs 1 --limit-train-batches 5 --limit-test-batches 2
```

Smoke test Shampoo with pseudoinverse root inverse:

```bash
uv run python train_cifar_mlp.py --optimizer shampoo_pinv --epochs 1 --limit-train-batches 5 --limit-test-batches 2
```

Smoke test default Shampoo:

```bash
uv run python train_cifar_mlp.py --optimizer shampoo_default --epochs 1 --limit-train-batches 5 --limit-test-batches 2
```

Smoke test one-sided Shampoo with pseudoinverse root inverse:

```bash
uv run python train_cifar_mlp.py --optimizer shampoo_pinv_one_sided --epochs 1 --limit-train-batches 5 --limit-test-batches 2
```

Longer comparison:

```bash
uv run python train_cifar_mlp.py --optimizer muon --epochs 10 --seed 0
uv run python train_cifar_mlp.py --optimizer shampoo_default --epochs 10 --seed 0
uv run python train_cifar_mlp.py --optimizer shampoo_pinv --epochs 10 --seed 0
uv run python train_cifar_mlp.py --optimizer shampoo_pinv_one_sided --epochs 10 --seed 0
```

Metrics are written as JSONL files under `logs/`.
