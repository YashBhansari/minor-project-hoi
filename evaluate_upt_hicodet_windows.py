"""Evaluate the official UPT HICO-DET checkpoint in one Windows process.

This keeps UPT's model, dataset, and HICO metric implementation. It bypasses
the training entrypoint's NCCL/DDP wrapper, which is unavailable on Windows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torchvision
from torch.utils.data import DataLoader, SequentialSampler


WORK_ROOT = Path(__file__).resolve().parent
UPT_ROOT = WORK_ROOT / "repos" / "upt"
sys.path.insert(0, str(UPT_ROOT))
os.chdir(UPT_ROOT)

# UPT's full checkpoint replaces the detector initialization. Avoid a separate
# ImageNet download while constructing the model; the loaded checkpoint is strict.
_resnet50 = torchvision.models.resnet50


def _resnet50_without_initial_weights(*args, **kwargs):
    kwargs.pop("pretrained", None)
    kwargs.pop("weights", None)
    return _resnet50(*args, weights=None, **kwargs)


torchvision.models.resnet50 = _resnet50_without_initial_weights

from utils import DataFactory, CustomisedDLE, custom_collate  # noqa: E402
from upt import build_detector  # noqa: E402


class ModelAdapter(torch.nn.Module):
    """Expose DDP's `.module` attribute while forwarding directly to UPT."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.module = model

    def forward(self, images):
        return self.module(images)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=WORK_ROOT / "data")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=WORK_ROOT / "checkpoints" / "official" / "upt-r50-hicodet.pt",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=WORK_ROOT / "outputs" / "hicodet",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to run an accidental CPU benchmark.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    torch.manual_seed(66)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)

    trainset = DataFactory("hicodet", "train2015", str(args.data_root))
    testset = DataFactory("hicodet", "test2015", str(args.data_root))
    loader = DataLoader(
        testset,
        batch_size=1,
        sampler=SequentialSampler(testset),
        num_workers=args.num_workers,
        collate_fn=custom_collate,
        pin_memory=True,
        drop_last=False,
    )

    model_args = SimpleNamespace(
        backbone="resnet50", dilation=False, position_embedding="sine",
        num_queries=100, aux_loss=True, set_cost_class=1, set_cost_bbox=5,
        set_cost_giou=2, bbox_loss_coef=5, giou_loss_coef=2, eos_coef=0.1,
        hidden_dim=256, enc_layers=6, dec_layers=6, dim_feedforward=2048,
        dropout=0.1, nheads=8, pre_norm=False, repr_dim=512, num_classes=117,
        human_idx=0, alpha=0.5, gamma=0.2, box_score_thresh=0.2,
        fg_iou_thresh=0.5, min_instances=3, max_instances=15,
        pretrained="", device="cuda",
    )
    model = build_detector(model_args, trainset.dataset.object_to_verb)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = ModelAdapter(model.cuda()).eval()

    # Reuse the authors' exact HICO-DET AP implementation without constructing
    # Pocket's DistributedDataParallel engine.
    engine = SimpleNamespace(_state=SimpleNamespace(net=model))
    start = time.perf_counter()
    ap = CustomisedDLE.test_hico(engine, loader)
    elapsed = time.perf_counter() - start

    train_counts = torch.as_tensor(trainset.dataset.anno_interaction)
    rare = train_counts < 10
    non_rare = train_counts >= 10
    metrics = {
        "model": "UPT-R50",
        "dataset": "HICO-DET",
        "partition": "test2015",
        "protocol": "Default",
        "num_evaluated_images": len(testset),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "compatibility_path": "single-process; official UPT test_hico metric method",
        "full_mAP": float(ap.mean()),
        "rare_mAP": float(ap[rare].mean()),
        "non_rare_mAP": float(ap[non_rare].mean()),
        "per_interaction_AP": [float(x) for x in ap.tolist()],
        "runtime_seconds": elapsed,
        "peak_cuda_allocated_MiB": torch.cuda.max_memory_allocated(0) / (1024**2),
        "peak_cuda_reserved_MiB": torch.cuda.max_memory_reserved(0) / (1024**2),
    }
    output_file = args.output_dir / "metrics.json"
    output_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in (
        "num_evaluated_images", "full_mAP", "rare_mAP", "non_rare_mAP",
        "runtime_seconds", "peak_cuda_allocated_MiB", "peak_cuda_reserved_MiB",
    )}, indent=2))
    print(f"Saved complete metrics and per-class AP: {output_file}")


if __name__ == "__main__":
    main()
