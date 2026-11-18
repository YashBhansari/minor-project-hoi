import argparse
import json
import time
from pathlib import Path

import torch

from common import (WORK_ROOT, CustomisedDLE, DataFactory, cuda_peak_memory, engine, environment,
                    load_eval_model, resolve, sha256, test_loader)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=WORK_ROOT / "data")
    parser.add_argument("--checkpoint", type=Path,
                        default=WORK_ROOT / "checkpoints" / "upt-r50-vcoco.pt")
    parser.add_argument("--output-dir", type=Path,
                        default=WORK_ROOT / "outputs" / "trained-evaluation" / "official-vcoco")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--trained-on", choices=("hicodet", "vcoco"), default="vcoco",
                        help="Dataset the checkpoint was trained on; its classes are "
                             "relabelled when it differs from the test dataset.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.data_root, args.checkpoint, args.output_dir = map(
        resolve, (args.data_root, args.checkpoint, args.output_dir))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to cache on CPU.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    torch.manual_seed(66)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)

    data_root = str(args.data_root)
    trainset = DataFactory("vcoco", "trainval", data_root)
    testset = DataFactory("vcoco", "test", data_root)
    source = trainset if args.trained_on == "vcoco" else DataFactory("hicodet", "train2015", data_root)
    model = load_eval_model(args.checkpoint, source, trainset)

    start = time.perf_counter()
    CustomisedDLE.cache_vcoco(engine(model), test_loader(testset, args.num_workers),
                              str(args.output_dir))
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
        "environment": environment(),
        "compatibility_path": "single-process; official UPT cache_vcoco method",
        "runtime_seconds": elapsed,
        **cuda_peak_memory(),
    }
    manifest_path = args.output_dir / "cache-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
