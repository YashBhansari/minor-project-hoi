# Experiment protocol: UPT-R50 on HICO-DET and V-COCO

## Objective

Run one UPT-R50 architecture in two independent dataset-specific experiments. Start each interaction head from its dataset's official DETR-R50 initialization, freeze the detector, fine-tune the interaction head, and evaluate the resulting final checkpoint on the matching test split.

## Hardware and runtime

Original machine: Windows 11, RTX 2050 with 4 GB VRAM, 16 GB RAM. Local software check: Python 3.14.7, PyTorch 2.14.1+cu130, torchvision 0.29.1+cu130. The checked-in environment freeze records the installed packages. The launcher path to Python is machine-specific and must be updated if running elsewhere.

## Training configuration in the launchers

| Setting | Value |
|---|---|
| Model | UPT-R50 |
| Runs | HICO-DET and V-COCO separately |
| Detector initialization | Official dataset-specific DETR-R50 `.pth` file |
| Trainable parameters | UPT interaction head; detector frozen |
| Epochs | 20 |
| Batch size | 16 per optimizer step, as in the paper; accumulated over forward passes of 2 images |
| Optimizer | AdamW |
| Interaction-head learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Learning-rate drop | Epoch 10 |
| Gradient clipping | 0.1 |
| Random seed | 66 |
| Checkpoint interval | Every 250 optimizer steps and each epoch boundary |
| Distributed training | Disabled; one process on Windows |

GPU memory is the reason for accumulating small forward passes in a single process. The trainer shows a live progress bar and does not create training logs. Check the checkpoint file's last-modified time and process CPU time to monitor activity. Do not claim a duration in advance; record actual runtime.

## Evaluation protocol

- HICO-DET: run `evaluate_upt_hicodet_windows.py` on `test2015`; report Default Full/Rare/Non-Rare mAP.
- V-COCO: generate the UPT prediction cache with `cache_upt_vcoco_windows.py`; score using the patched Gupta evaluator via `evaluate_upt_vcoco_cache.py`; report AP_role for Scenarios 1 and 2 and state the UPT 25/24 correction.
- Commands and output paths are in [HANDOFF.md](HANDOFF.md).

## Dataset loader sizes

The UPT data loaders filter to images with annotated interactions; counts below are not raw image-folder totals.

| Dataset partition | UPT loader images |
|---|---:|
| HICO-DET train2015 | 37,633 |
| HICO-DET test2015 | 9,546 |
| V-COCO trainval | 4,969 |
| V-COCO test | 4,532 |

## Reference metrics

| Dataset / metric | UPT-R50 paper reference |
|---|---:|
| HICO-DET Default Full / Rare / Non-Rare mAP | 31.66 / 25.94 / 33.36 |
| V-COCO AP_role Scenario 1 / Scenario 2 | 59.0 / 64.5 |

These are comparison references, not expected or guaranteed local training outcomes. The separately measured released-checkpoint results are documented in [results.md](results.md).

## Reproduction deviations and code changes

1. Training uses single-process Windows CUDA instead of the upstream distributed setup.
2. The paper's 16-image step is reproduced by gradient accumulation: each forward pass back-propagates its un-normalised loss, and the accumulated gradient is divided by the positive matches of all 16 images, which is the update the 8-GPU recipe computes.
3. The local UPT interaction loss clamps the positive-match denominator to at least one so a batch with no positive matches does not yield division by zero.
4. The V-COCO evaluator was patched for current Python and NumPy compatibility and indexes detections by image for evaluation speed.
5. Evaluation wrappers call the upstream metric/cache methods directly to avoid UPT's Windows-incompatible distributed entry point.

Details and diffs: [PATCHES.md](PATCHES.md), `vcoco-evaluator-python3.patch`.

## Result integrity

Keep three things separate: paper reference numbers, evaluation numbers from official released UPT checkpoints, and measurements from the local fine-tuned checkpoints. Only the last category answers how the local training runs performed. Record each locally trained result from its evaluation JSON; do not fill a gap by copying a paper or baseline value.
