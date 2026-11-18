import argparse
import json
import sys
import time
from pathlib import Path

import torch

from common import (WORK_ROOT, CustomisedDLE, DataFactory, cuda_peak_memory, engine, environment,
                    load_eval_model, resolve, sha256, test_loader)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=WORK_ROOT / "data")
    parser.add_argument("--checkpoint", type=Path,
                        default=WORK_ROOT / "checkpoints" / "upt-r50-hicodet.pt")
    parser.add_argument("--output-dir", type=Path,
                        default=WORK_ROOT / "outputs" / "trained-evaluation" / "official-hico")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--trained-on", choices=("hicodet", "vcoco"), default="hicodet",
                        help="Dataset the checkpoint was trained on; its classes are "
                             "relabelled when it differs from the test dataset.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.data_root, args.checkpoint, args.output_dir = map(
        resolve, (args.data_root, args.checkpoint, args.output_dir))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to evaluate on CPU.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    torch.manual_seed(66)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)

    data_root = str(args.data_root)
    trainset = DataFactory("hicodet", "train2015", data_root)
    testset = DataFactory("hicodet", "test2015", data_root)
    source = trainset if args.trained_on == "hicodet" else DataFactory("vcoco", "trainval", data_root)
    model = load_eval_model(args.checkpoint, source, trainset)

    start = time.perf_counter()
    ap = CustomisedDLE.test_hico(engine(model), test_loader(testset, args.num_workers))
    elapsed = time.perf_counter() - start

    train_counts = torch.as_tensor(trainset.dataset.anno_interaction)
    metrics = {
        "model": "UPT-R50",
        "dataset": "HICO-DET",
        "partition": "test2015",
        "protocol": "Default",
        "num_evaluated_images": len(testset),
        "checkpoint": str(args.checkpoint),
        "trained_on": args.trained_on,
        "checkpoint_sha256": sha256(args.checkpoint),
        "environment": environment(),
        "compatibility_path": "single-process; official UPT test_hico metric method",
        "full_mAP": float(ap.mean()),
        "rare_mAP": float(ap[train_counts < 10].mean()),
        "non_rare_mAP": float(ap[train_counts >= 10].mean()),
        "per_interaction_AP": [float(x) for x in ap.tolist()],
        "runtime_seconds": elapsed,
        **cuda_peak_memory(),
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
