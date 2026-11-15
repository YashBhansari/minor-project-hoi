"""Single-GPU Windows trainer for the official UPT-R50 training recipe.

The dataset-specific DETR detector is initialized from the official UPT model
zoo. As in upstream UPT, detector parameters are frozen and only the HOI
interaction head is optimized. This bypasses DDP because this Windows build
cannot safely wrap CUDA models with its available backend.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import os
import random
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torchvision
import torch.distributed as dist
import pocket
from torch.utils.data import DataLoader, Sampler

warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")

WORK_ROOT = Path(__file__).resolve().parent
UPT_ROOT = WORK_ROOT / "repos" / "upt"
sys.path.insert(0, str(UPT_ROOT))
os.chdir(UPT_ROOT)

# UPT's checkpoint supplies these weights; avoid a separate ImageNet download.
_resnet50 = torchvision.models.resnet50


def _resnet50_without_initial_weights(*args, **kwargs):
    kwargs.pop("pretrained", None)
    kwargs.pop("weights", None)
    return _resnet50(*args, weights=None, **kwargs)


torchvision.models.resnet50 = _resnet50_without_initial_weights

from utils import DataFactory, custom_collate  # noqa: E402
from upt import build_detector  # noqa: E402


class ResumableRandomBatchSampler(Sampler[list[int]]):
    """Random epoch order with a restorable next-batch offset."""

    def __init__(self, size: int, batch_size: int, seed: int):
        self.size = size
        self.batch_size = batch_size
        self.seed = seed
        self.permutation: list[int] = []
        self.offset = 0

    def set_epoch(self, epoch: int, permutation=None, offset: int = 0):
        if permutation is None:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            self.permutation = torch.randperm(self.size, generator=generator).tolist()
        else:
            self.permutation = [int(index) for index in permutation]
        self.offset = int(offset)

    def __iter__(self):
        limit = self.size - (self.size % self.batch_size)
        while self.offset + self.batch_size <= limit:
            begin = self.offset
            self.offset += self.batch_size
            yield self.permutation[begin:self.offset]

    def __len__(self):
        return max(0, (self.size - self.offset) // self.batch_size)


def rng_state():
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(0),
    }


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np_state = state["numpy"]
    np.random.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32), *np_state[2:]))
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"], 0)


def atomic_save(payload, destination: Path):
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("hicodet", "vcoco"), required=True)
    parser.add_argument("--data-root", type=Path, default=WORK_ROOT / "data")
    parser.add_argument("--detr-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-drop", type=int, default=10)
    parser.add_argument("--clip-max-norm", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Stop after this many optimizer steps (smoke run).")
    parser.add_argument("--checkpoint-interval", type=int, default=250,
                        help="Save resumable state after this many optimizer steps.")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume from a checkpoint written by this script.")
    return parser.parse_args()


def main():
    args = get_args()
    def absolute_from_work_root(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (WORK_ROOT.parent / path).resolve()

    args.data_root = absolute_from_work_root(args.data_root)
    args.detr_checkpoint = absolute_from_work_root(args.detr_checkpoint)
    args.output_dir = absolute_from_work_root(args.output_dir)
    if args.resume is not None:
        args.resume = absolute_from_work_root(args.resume)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to start training on CPU.")
    if not args.detr_checkpoint.is_file():
        raise FileNotFoundError(args.detr_checkpoint)
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(args.resume)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(0)

    train_partition = "train2015" if args.dataset == "hicodet" else "trainval"
    dataset = DataFactory(args.dataset, train_partition, str(args.data_root))
    if args.checkpoint_interval < 1:
        raise ValueError("--checkpoint-interval must be at least 1.")
    batch_sampler = ResumableRandomBatchSampler(len(dataset), args.batch_size, args.seed)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0,
                        collate_fn=custom_collate, pin_memory=True,
                        generator=torch.Generator().manual_seed(args.seed))

    is_hico = args.dataset == "hicodet"
    num_classes = 117 if is_hico else 24
    class_corr = (
        dataset.dataset.object_to_verb if is_hico
        else list(dataset.dataset.object_to_action.values())
    )
    model_args = SimpleNamespace(
        backbone="resnet50", dilation=False, position_embedding="sine",
        num_queries=100, aux_loss=True, set_cost_class=1, set_cost_bbox=5,
        set_cost_giou=2, bbox_loss_coef=5, giou_loss_coef=2, eos_coef=0.1,
        hidden_dim=256, enc_layers=6, dec_layers=6, dim_feedforward=2048,
        dropout=0.1, nheads=8, pre_norm=False, repr_dim=512,
        num_classes=num_classes, human_idx=0, alpha=0.5, gamma=0.2,
        box_score_thresh=0.2, fg_iou_thresh=0.5, min_instances=3,
        max_instances=15, pretrained=str(args.detr_checkpoint), device="cuda",
    )
    # UPT's builder calls get_rank only to print a message when loading the
    # detector weights. Avoid initializing Gloo: training itself is fully
    # single-process and does not use any distributed collective.
    original_get_rank = dist.get_rank
    if not dist.is_initialized():
        dist.get_rank = lambda group=None: 0
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            model = build_detector(model_args, class_corr).cuda()
    finally:
        dist.get_rank = original_get_rank
    checkpoint = None
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        start_epoch = int(checkpoint["next_epoch"] if checkpoint["epoch_in_progress"] is None
                          else checkpoint["epoch_in_progress"])
        resume_in_progress = checkpoint["epoch_in_progress"] is not None
        resume_offset = int(checkpoint["batch_offset"])
        iteration = int(checkpoint["iteration"])
    else:
        start_epoch = 0
        resume_in_progress = False
        resume_offset = 0
        iteration = 0

    # Match upstream UPT: keep DETR fixed; train the interaction head only.
    for parameter in model.detector.parameters():
        parameter.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr_head, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    if checkpoint is not None and "optim_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optim_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    metadata = {
        "dataset": args.dataset,
        "partition": train_partition,
        "train_images": len(dataset),
        "initialization": str(args.detr_checkpoint),
        "initialization_sha256": sha256(args.detr_checkpoint),
        "initialization_type": "official dataset-specific DETR-R50 checkpoint",
        "model": "UPT-R50",
        "training_mode": "single GPU; frozen detector; train interaction head",
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate_head": args.lr_head,
        "seed": args.seed,
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "python": sys.version.split()[0],
    }
    torch.cuda.reset_peak_memory_stats(0)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    steps_this_run = 0
    latest_path = args.output_dir / "checkpoint_latest.pt"

    def make_checkpoint(epoch_in_progress, next_epoch, batch_offset, permutation):
        return {
            "epoch_in_progress": epoch_in_progress,
            "next_epoch": next_epoch,
            "batch_offset": batch_offset,
            "sampler_permutation": None if permutation is None else torch.as_tensor(permutation),
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "rng_state": rng_state(),
            "run": metadata,
        }

    if checkpoint is not None:
        restore_rng_state(checkpoint.get("rng_state"))

    try:
        for epoch in range(start_epoch, args.epochs):
            is_resumed_epoch = epoch == start_epoch and resume_in_progress
            batch_sampler.set_epoch(
                epoch,
                checkpoint["sampler_permutation"] if is_resumed_epoch else None,
                resume_offset if is_resumed_epoch else 0,
            )
            resume_in_progress = False
            checkpoint = None
            model.train()
            for batch in loader:
                images = pocket.ops.relocate_to_cuda(batch[0], non_blocking=True)
                targets = pocket.ops.relocate_to_cuda(batch[1], non_blocking=True)
                loss_dict = model(images, targets=targets)
                loss = sum(loss_dict.values())
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at iteration {iteration}.")
                loss.backward()
                if args.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, args.clip_max_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                iteration += 1
                steps_this_run += 1
                if iteration % args.checkpoint_interval == 0:
                    atomic_save(make_checkpoint(epoch, epoch + 1, batch_sampler.offset,
                                                batch_sampler.permutation), latest_path)
                if args.max_steps and steps_this_run >= args.max_steps:
                    atomic_save(make_checkpoint(epoch, epoch + 1, batch_sampler.offset,
                                                batch_sampler.permutation), latest_path)
                    return

            scheduler.step()
            atomic_save(make_checkpoint(None, epoch + 1, 0, None), latest_path)

        final_path = args.output_dir / "checkpoint_final.pt"
        if latest_path.exists():
            os.replace(latest_path, final_path)

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise


if __name__ == "__main__":
    main()
