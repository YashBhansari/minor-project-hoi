"""Cache official UPT V-COCO predictions in one Windows process."""

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

# The UPT checkpoint contains detector weights, so do not fetch separate
# torchvision ResNet initialization weights before strictly loading it.
_resnet50 = torchvision.models.resnet50


def _resnet50_without_initial_weights(*args, **kwargs):
    kwargs.pop("pretrained", None)
    kwargs.pop("weights", None)
    return _resnet50(*args, weights=None, **kwargs)


torchvision.models.resnet50 = _resnet50_without_initial_weights

from utils import DataFactory, CustomisedDLE, custom_collate  # noqa: E402
from upt import build_detector  # noqa: E402
import cross_dataset  # noqa: E402


class ModelAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.module = model

    def forward(self, images):
        return self.module(images)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=WORK_ROOT / "data")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=WORK_ROOT / "checkpoints" / "official" / "upt-r50-vcoco.pt",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=WORK_ROOT / "outputs" / "vcoco",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--trained-on", choices=("hicodet", "vcoco"), default="vcoco",
                        help="Dataset the checkpoint was trained on; its classes are "
                             "relabelled when it differs from the test dataset.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.data_root = (WORK_ROOT / args.data_root).resolve()
    args.checkpoint = (WORK_ROOT / args.checkpoint).resolve()
    args.output_dir = (WORK_ROOT / args.output_dir).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to cache on CPU accidentally.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    torch.manual_seed(66)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)

    trainset = DataFactory("vcoco", "trainval", str(args.data_root))
    testset = DataFactory("vcoco", "test", str(args.data_root))
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
        dropout=0.1, nheads=8, pre_norm=False, repr_dim=512, num_classes=24,
        human_idx=0, alpha=0.5, gamma=0.2, box_score_thresh=0.2,
        fg_iou_thresh=0.5, min_instances=3, max_instances=15,
        pretrained="", device="cuda",
    )
    source = trainset if args.trained_on == "vcoco" else DataFactory(
        "hicodet", "train2015", str(args.data_root))
    names, class_corr = cross_dataset.classes(source.dataset)
    model_args.num_classes = len(names)
    model = build_detector(model_args, class_corr)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.cuda()
    model = (ModelAdapter(model) if source is trainset else
             cross_dataset.Relabel(model, source.dataset, trainset.dataset).cuda()).eval()

    # Reuse the authors' V-COCO prediction formatter and cache implementation.
    engine = SimpleNamespace(_state=SimpleNamespace(net=model))
    start = time.perf_counter()
    CustomisedDLE.cache_vcoco(engine, loader, str(args.output_dir))
    elapsed = time.perf_counter() - start

    cache_path = args.output_dir / "cache.pkl"
    manifest = {
        "model": "UPT-R50",
        "dataset": "V-COCO",
        "partition": "test",
        "num_evaluated_images": len(testset),
        "checkpoint": str(args.checkpoint),
        "trained_on": args.trained_on,
        "checkpoint_sha256": sha256(args.checkpoint),
        "cache": str(cache_path),
        "cache_sha256": sha256(cache_path),
        "cache_bytes": cache_path.stat().st_size,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "compatibility_path": "single-process; official UPT cache_vcoco method",
        "runtime_seconds": elapsed,
        "peak_cuda_allocated_MiB": torch.cuda.max_memory_allocated(0) / (1024**2),
        "peak_cuda_reserved_MiB": torch.cuda.max_memory_reserved(0) / (1024**2),
    }
    manifest_path = args.output_dir / "cache-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
