"""Score a UPT V-COCO cache with the official Gupta evaluator."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import argparse
from contextlib import redirect_stdout
from pathlib import Path

WORK_ROOT = Path(__file__).resolve().parent
EVALUATOR_ROOT = WORK_ROOT / "repos" / "v-coco"
DATA_ROOT = WORK_ROOT / "data"
ANNOTATIONS_ROOT = DATA_ROOT / "mscoco2014" / "annotations"
CACHE_PATH = WORK_ROOT / "outputs" / "vcoco" / "cache.pkl"
OUTPUT_DIR = WORK_ROOT / "outputs" / "vcoco"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    cache_path = args.cache.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not cache_path.is_file():
        raise FileNotFoundError(f"UPT V-COCO cache is missing: {cache_path}")
    # script_pick_annotations.py writes the filtered file into the evaluator's
    # own data directory, rather than beside the source COCO annotations.
    all_coco_annotations = EVALUATOR_ROOT / "data" / "instances_vcoco_all_2014.json"
    if not all_coco_annotations.is_file():
        subprocess.run(
            [
                sys.executable,
                str(EVALUATOR_ROOT / "script_pick_annotations.py"),
                str(ANNOTATIONS_ROOT),
            ],
            check=True,
            cwd=EVALUATOR_ROOT,
        )

    sys.path.insert(0, str(EVALUATOR_ROOT))
    os.chdir(EVALUATOR_ROOT)
    from vsrl_eval import VCOCOeval

    evaluator = VCOCOeval(
        str(EVALUATOR_ROOT / "data" / "vcoco" / "vcoco_test.json"),
        str(all_coco_annotations),
        str(EVALUATOR_ROOT / "data" / "splits" / "vcoco_test.ids"),
    )
    capture = io.StringIO()
    with redirect_stdout(capture):
        evaluator._do_eval(str(cache_path), ovr_thresh=0.5)
    report = capture.getvalue()
    print(report)

    raw_values = {}
    for scenario in ("scenario_1", "scenario_2"):
        match = re.search(
            rf"Average Role \[{scenario}\] AP = ([0-9.]+)", report
        )
        if match is None:
            raise RuntimeError(f"Could not find {scenario} role AP in evaluator output.")
        raw_values[scenario] = float(match.group(1))

    # UPT authors specify that the point-instrument class is excluded from
    # the final V-COCO role score by multiplying the evaluator mean by 25/24.
    metrics = {
        "dataset": "V-COCO",
        "model": "UPT-R50",
        "evaluator": "Gupta et al. official V-COCO evaluator",
        "evaluator_commit": "489cc4db74f2f10ab4b134f67da3874afbf245ab",
        "overlap_threshold": 0.5,
        "upt_cache": str(cache_path),
        "upt_cache_sha256": sha256(cache_path),
        "coco_annotation_subset": str(all_coco_annotations),
        "scenarios_raw_AP_percent": raw_values,
        "scenarios_upt_corrected_AP_percent": {
            key: value * 25.0 / 24.0 for key, value in raw_values.items()
        },
        "correction": "multiply by 25/24 per UPT authors' V-COCO evaluation instructions",
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved V-COCO metrics to {metrics_path}")


if __name__ == "__main__":
    main()
