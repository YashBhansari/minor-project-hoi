import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

WORK_ROOT = Path(__file__).resolve().parent
EVALUATOR_ROOT = WORK_ROOT / "repos" / "v-coco"
ANNOTATIONS_ROOT = WORK_ROOT / "data" / "mscoco2014" / "annotations"
DEFAULT_DIR = WORK_ROOT / "outputs" / "trained-evaluation" / "official-vcoco"
ROW = re.compile(r"^\s*(\S+): AP = ([0-9.]+) \(#pos = \d+\)", re.M)


def ap_rows(block):
    return {m.group(1): float(m.group(2)) for m in ROW.finditer(block)}


def per_class_ap(report):
    agent_end = report.index("Average Agent AP")
    role_start = report.index("Reporting Role AP")
    s1_end = report.index("Average Role [scenario_1]")
    s2_end = report.index("Average Role [scenario_2]")
    return {
        "agent_AP_percent": ap_rows(report[:agent_end]),
        "role_AP_percent": {
            "scenario_1": ap_rows(report[role_start:s1_end]),
            "scenario_2": ap_rows(report[s1_end:s2_end]),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_DIR / "cache.pkl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    cache_path = args.cache.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not cache_path.is_file():
        raise FileNotFoundError(f"UPT V-COCO cache is missing: {cache_path}")

    all_coco_annotations = EVALUATOR_ROOT / "data" / "instances_vcoco_all_2014.json"
    if not all_coco_annotations.is_file():
        subprocess.run([sys.executable, str(EVALUATOR_ROOT / "script_pick_annotations.py"),
                        str(ANNOTATIONS_ROOT)], check=True, cwd=EVALUATOR_ROOT)

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
        match = re.search(rf"Average Role \[{scenario}\] AP = ([0-9.]+)", report)
        if match is None:
            raise RuntimeError(f"Could not find {scenario} role AP in evaluator output.")
        raw_values[scenario] = float(match.group(1))

    metrics = {
        "dataset": "V-COCO",
        "model": "UPT-R50",
        "evaluator": "Gupta et al. official V-COCO evaluator",
        "evaluator_commit": "489cc4db74f2f10ab4b134f67da3874afbf245ab",
        "overlap_threshold": 0.5,
        "upt_cache": str(cache_path),
        "upt_cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        "coco_annotation_subset": str(all_coco_annotations),
        "scenarios_raw_AP_percent": raw_values,
        "scenarios_upt_corrected_AP_percent": {
            key: value * 25.0 / 24.0 for key, value in raw_values.items()
        },
        "correction": "multiply by 25/24 per UPT authors' V-COCO evaluation instructions",
        **per_class_ap(report),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved V-COCO metrics to {metrics_path}")


if __name__ == "__main__":
    main()
