# Upstream compatibility patches

## Official Gupta V-COCO evaluator compatibility patch

- Repository: `work/repos/v-coco`
- Upstream commit: `489cc4db74f2f10ab4b134f67da3874afbf245ab`
- Files: `vsrl_utils.py`, `vsrl_eval.py`, plus a compatibility helper in `utils.py`
- Reason: Python 2 names (`xrange`, `unicode`, `dict.iteritems`) and NumPy aliases removed in NumPy 2 prevent the official evaluator from running under the project's Python 3.14 / NumPy 2.5 environment.
- Change: use `range`, `dict.items`, retain Python 3 JSON strings, and replace removed NumPy aliases (`np.NaN`, `np.float`, `np.bool`) with supported equivalents.
- Full unified diff: [vcoco-evaluator-python3.patch](vcoco-evaluator-python3.patch)
- The evaluator uses the `pycocotools` package; no Cython bbox extension is needed for its evaluation path. The Windows-compatible `pycocotools` wheel is installed in the isolated project environment.
- Added `work/repos/v-coco/utils.py` with UPT's documented `CacheTemplate` so Python can unpickle the UPT cache and use zero values for missing action/role keys.
- `vsrl_eval.py` also converts NumPy's one-element `where` result to a scalar explicitly (required by current NumPy) and indexes cached detections by image once. The index preserves prediction order and matching logic while avoiding a full cache scan for every test image.
- The filtered COCO annotation file is generated at `work/repos/v-coco/data/instances_vcoco_all_2014.json` by the official `script_pick_annotations.py`; the project wrapper points the evaluator to that output location.

## UPT Windows evaluation wrapper

- No UPT repository files have been modified.
- `evaluate_upt_hicodet_windows.py` and `cache_upt_vcoco_windows.py` reuse the official UPT metric/cache methods while bypassing the `main.py` NCCL/DDP wrapper. Windows Gloo initialized, but a CUDA DDP forward/backward smoke check crashed with access violation, so direct single-process execution is used.
- Both wrappers skip loading separate torchvision ResNet initialization weights because the strict UPT checkpoint already contains the detector weights. This only avoids an unnecessary network download; model weights are then loaded strictly from the official checkpoint.

## UPT single-GPU training adaptation

- Repository: `work/repos/upt/upt.py`.
- Change: in `compute_interaction_loss`, divide by `max(n_p, 1)` instead of `n_p` directly. The official distributed recipe aggregates positive matches across workers; a one-image, single-GPU batch can contain no positive matches, which otherwise makes loss `0/0` and aborts training. With no positive matches, this keeps the all-negative batch loss finite.
- The patch also stores the positive-match count as `self.num_positives`, so the trainer can normalise a 16-image step by the positives of all its images, as the distributed recipe does.
- The detector remains frozen and only the interaction head is optimized, as in the upstream design. Training uses one process and accumulates gradients over small forward passes because this Windows CUDA build cannot safely use UPT's DDP wrapper.
- The launcher stores the optimizer, scheduler, sample order, random-number-generator state, and model weights in the same resumable checkpoint. No per-step or run log files are created.

## Patch files for a fresh clone

- `upt-single-gpu-loss.patch` contains the one-line safe-denominator change to the pinned UPT commit `192743045e4c5f6827299fdc15809f5b97ac1b8e`.
- `vcoco-evaluator-python3.patch` contains the V-COCO evaluator compatibility edits and the added `utils.py` cache-unpickle helper for commit `489cc4db74f2f10ab4b134f67da3874afbf245ab`.
- Apply these from the project root only after checking out those exact commits. The commands are in [HANDOFF.md](HANDOFF.md). Do not apply either patch twice.
