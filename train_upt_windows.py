import argparse
import contextlib
import io
import os
import random
import sys
from pathlib import Path

import numpy as np
import pocket
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

import cross_dataset
from common import WORK_ROOT, DataFactory, build_detector, custom_collate, environment, model_args, resolve, sha256


class ResumableRandomBatchSampler(Sampler):
    def __init__(self, size, batch_size, seed):
        self.size = size
        self.batch_size = batch_size
        self.seed = seed
        self.permutation = []
        self.offset = 0

    def set_epoch(self, epoch, permutation=None, offset=0):
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
    random.setstate(state["python"])
    np_state = state["numpy"]
    np.random.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32), *np_state[2:]))
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"], 0)


def atomic_save(payload, destination):
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("hicodet", "vcoco"), required=True)
    parser.add_argument("--data-root", type=Path, default=WORK_ROOT / "data")
    parser.add_argument("--detr-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Images per optimizer step (the paper uses 16).")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Processes that load and augment images in parallel.")
    parser.add_argument("--micro-batch", type=int, default=2,
                        help="Images per forward pass; gradients are accumulated over "
                             "batch-size / micro-batch passes to fit in GPU memory.")
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
    args.data_root = resolve(args.data_root)
    args.detr_checkpoint = resolve(args.detr_checkpoint)
    args.output_dir = resolve(args.output_dir)
    if args.resume is not None:
        args.resume = resolve(args.resume)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to start training on CPU.")
    if not args.detr_checkpoint.is_file():
        raise FileNotFoundError(args.detr_checkpoint)
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(args.resume)
    if args.checkpoint_interval < 1:
        raise ValueError("--checkpoint-interval must be at least 1.")
    if args.batch_size % args.micro_batch:
        raise ValueError("--batch-size must be a multiple of --micro-batch.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(0)

    print(f"Loading {args.dataset} and the model; the progress bar appears in under a minute.", flush=True)
    train_partition = "train2015" if args.dataset == "hicodet" else "trainval"
    dataset = DataFactory(args.dataset, train_partition, str(args.data_root))
    accumulate = args.batch_size // args.micro_batch
    batch_sampler = ResumableRandomBatchSampler(len(dataset), args.micro_batch, args.seed)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=args.num_workers,
                        collate_fn=custom_collate, pin_memory=True,
                        generator=torch.Generator().manual_seed(args.seed))

    names, class_corr = cross_dataset.classes(dataset.dataset)
    original_get_rank = dist.get_rank
    dist.get_rank = lambda group=None: 0
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            model = build_detector(model_args(len(names), str(args.detr_checkpoint)), class_corr).cuda()
    finally:
        dist.get_rank = original_get_rank

    checkpoint = None
    start_epoch = resume_offset = iteration = 0
    resume_in_progress = False
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        resume_in_progress = checkpoint["epoch_in_progress"] is not None
        start_epoch = int(checkpoint["epoch_in_progress"] if resume_in_progress
                          else checkpoint["next_epoch"])
        resume_offset = int(checkpoint["batch_offset"])
        iteration = int(checkpoint["iteration"])
        run = checkpoint["run"]
        if run.get("micro_batch") != args.micro_batch or run["batch_size"] != args.batch_size:
            raise ValueError(f"{args.resume} was written with a different batch size; "
                             "move it away to start a new run.")

    for parameter in model.detector.parameters():
        parameter.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr_head, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optim_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        restore_rng_state(checkpoint["rng_state"])

    metadata = {
        "dataset": args.dataset,
        "partition": train_partition,
        "train_images": len(dataset),
        "initialization": str(args.detr_checkpoint),
        "initialization_sha256": sha256(args.detr_checkpoint),
        "initialization_type": "official dataset-specific DETR-R50 checkpoint",
        "model": "UPT-R50",
        "training_mode": "single GPU; frozen detector; train interaction head; gradient accumulation",
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "micro_batch": args.micro_batch,
        "learning_rate_head": args.lr_head,
        "seed": args.seed,
        **environment(),
    }
    latest_path = args.output_dir / "checkpoint_latest.pt"

    def save_latest(epoch_in_progress, next_epoch, batch_offset, permutation):
        atomic_save({
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
        }, latest_path)

    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=args.epochs * (len(dataset) // args.batch_size), initial=iteration,
                    unit="step", file=sys.stdout, dynamic_ncols=True, smoothing=0.05)
    steps_this_run = passes = positives = raw_loss = 0
    average_loss = None

    for epoch in range(start_epoch, args.epochs):
        resumed = epoch == start_epoch and resume_in_progress
        batch_sampler.set_epoch(epoch,
                                checkpoint["sampler_permutation"] if resumed else None,
                                resume_offset if resumed else 0)
        model.train()
        consumed = batch_sampler.offset
        epoch_loss = epoch_steps = 0
        for batch in loader:
            consumed += args.micro_batch
            images = pocket.ops.relocate_to_cuda(batch[0], non_blocking=True)
            targets = pocket.ops.relocate_to_cuda(batch[1], non_blocking=True)
            loss = sum(model(images, targets=targets).values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at iteration {iteration}.")
            weight = max(model.num_positives, 1)
            (loss * weight).backward()
            positives += model.num_positives
            raw_loss += loss.item() * weight
            passes += 1
            if passes % accumulate:
                continue

            for parameter in trainable:
                if parameter.grad is not None:
                    parameter.grad /= max(positives, 1)
            step_loss = raw_loss / max(positives, 1)
            positives = raw_loss = 0
            epoch_loss += step_loss
            epoch_steps += 1
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.clip_max_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            iteration += 1
            steps_this_run += 1
            progress.update(1)
            average_loss = step_loss if average_loss is None else 0.99 * average_loss + 0.01 * step_loss
            progress.set_postfix(epoch=f"{epoch + 1}/{args.epochs}", avg_loss=f"{average_loss:.3f}")
            stop = args.max_steps and steps_this_run >= args.max_steps
            if iteration % args.checkpoint_interval == 0 or stop:
                save_latest(epoch, epoch + 1, consumed, batch_sampler.permutation)
            if stop:
                return

        optimizer.zero_grad(set_to_none=True)
        positives = passes = raw_loss = 0
        progress.write(f"epoch {epoch + 1}/{args.epochs} mean loss "
                       f"{epoch_loss / max(epoch_steps, 1):.4f} over {epoch_steps} steps")
        scheduler.step()
        save_latest(None, epoch + 1, 0, None)

    if latest_path.exists():
        os.replace(latest_path, args.output_dir / "checkpoint_final.pt")


if __name__ == "__main__":
    main()
