# Reproduction guide

How to set up the project, train both models, evaluate them, and rebuild the figures. Results are in [results.md](results.md).

## 1. What the project does

Human-object interaction (HOI) detection finds a person, an object, and the action linking them, for example a person riding a bicycle.

The project trains **UPT-R50**, the Unary-Pairwise Transformer with a ResNet-50 detector, on two datasets in two independent runs:

1. **HICO-DET**: train, then evaluate on `test2015`.
2. **V-COCO**: train, then evaluate on the test split.

The datasets have different action labels, so each run starts from its own dataset-specific DETR-R50 weights. UPT freezes the DETR detector and trains the interaction head. Each trained model is also evaluated on the other dataset by translating its predicted classes (see [results.md](results.md), section 3).

## 2. Machine and software

Results in this repository were produced on Windows 11 with an NVIDIA GeForce RTX 5060 Laptop GPU (8 GB), 16 GB RAM, Python 3.14.6, PyTorch 2.14.1+cu130 and torchvision 0.29.1+cu130. Installed packages are listed in `environment/windows-py314-freeze.txt`.

Everything runs as one process with 16 images per optimizer step (the paper's batch size), accumulated over forward passes of 2 images. Run one training job at a time.

Activate the virtual environment before running anything, or set `$env:UPT_PYTHON` to the full path of your `python.exe`. The PowerShell launchers use `UPT_PYTHON` if it is set and otherwise the `python` on `PATH`.

## 3. Folder map

| Path | Contents |
|---|---|
| `repos/upt/` | Official UPT source (pinned commit, with one patch) |
| `repos/v-coco/` | Gupta et al.'s V-COCO role evaluator (pinned commit, with compatibility patch) |
| `data/` | HICO-DET images and annotations, COCO 2014 images, UPT annotation JSON files |
| `checkpoints/official/` | DETR-R50 initializations and the authors' released UPT-R50 checkpoints |
| `outputs/training/<dataset>/` | Training checkpoints written by the launchers |
| `outputs/trained-evaluation/` | One folder per evaluation (table below) |
| `docs/figures/` | Figures used in the README and results |
| `train_hicodet.ps1`, `train_vcoco.ps1` | Training launchers |
| `train_upt_windows.py` | Single-process trainer |
| `evaluate_upt_hicodet_windows.py` | HICO-DET evaluation |
| `cache_upt_vcoco_windows.py`, `evaluate_upt_vcoco_cache.py` | V-COCO prediction cache and scoring |
| `cross_dataset.py` | Class translation used for cross-dataset evaluation |
| `make_figures.py` | Rebuilds `docs/figures/` from the metrics files |

`data/`, `checkpoints/official/` and `outputs/training/` are excluded from version control because of their size. Obtain the data from the sources in section 11 and keep the folder structure described in `data/manifest.md`.

Evaluation folders in `outputs/trained-evaluation/`:

| Folder | Model | Tested on |
|---|---|---|
| `hico-on-hico` | trained on HICO-DET | HICO-DET |
| `hico-on-vcoco` | trained on HICO-DET | V-COCO |
| `vcoco-on-vcoco` | trained on V-COCO | V-COCO |
| `vcoco-on-hico` | trained on V-COCO | HICO-DET |
| `official-hico` | authors' released HICO-DET checkpoint | HICO-DET |
| `official-hico-on-vcoco` | authors' released HICO-DET checkpoint | V-COCO |
| `official-vcoco` | authors' released V-COCO checkpoint | V-COCO |

## 4. Restore the upstream source checkouts

The upstream clones in `repos/` are not stored in this repository. Recreate the exact versions from PowerShell in the project folder and apply the checked-in patches:

```powershell
New-Item -ItemType Directory -Force .\repos | Out-Null

git clone --recurse-submodules https://github.com/fredzzhang/upt.git .\repos\upt
git -C .\repos\upt checkout 192743045e4c5f6827299fdc15809f5b97ac1b8e
git -C .\repos\upt submodule update --init --recursive
git -C .\repos\upt apply ..\..\upt-single-gpu-loss.patch

git clone https://github.com/s-gupta/v-coco.git .\repos\v-coco
git -C .\repos\v-coco checkout 489cc4db74f2f10ab4b134f67da3874afbf245ab
git -C .\repos\v-coco apply ..\..\vcoco-evaluator-python3.patch
```

`pocket` is installed as a Python dependency; its pinned revision is in `environment/windows-py314-freeze.txt`. Apply each patch once. If a patch fails, check that the upstream revision matches before retrying. What the patches change is described in [PATCHES.md](PATCHES.md).

## 5. Preflight

```powershell
Test-Path .\data\hico_20160224_det\images\train2015
Test-Path .\data\hico_20160224_det\images\test2015
Test-Path .\data\mscoco2014\train2014
Test-Path .\data\mscoco2014\val2014
Test-Path .\checkpoints\official\detr-r50-hicodet.pth
Test-Path .\checkpoints\official\detr-r50-vcoco.pth
python -c "import torch; print(torch.cuda.is_available())"
```

Every line should print `True`. Loader sizes for each partition are in `data/manifest.md`.

## 6. Train

Keep the machine on power and prevent sleep. Run one dataset at a time:

```powershell
.\train_hicodet.ps1
.\train_vcoco.ps1
```

Each launcher trains for 20 epochs with a batch of 16, saves a checkpoint every 250 optimizer steps, and resumes from `checkpoint_latest.pt` if interrupted. When training finishes the latest checkpoint is renamed `checkpoint_final.pt` in `outputs\training\<dataset>\`. Do not start two jobs for the same dataset, and do not delete `checkpoint_latest.pt` during a run.

The trainer shows a progress bar with the step, epoch, running average loss and time left, and does not write a log file. To monitor a run from a second window:

```powershell
Get-ChildItem .\outputs\training\hicodet -Force | Select-Object Name,Length,LastWriteTime
```

HICO-DET training took 47,040 steps (16 h 15 min). V-COCO training took 6,200 steps.

## 7. Evaluate

Run these once `checkpoint_final.pt` exists. Each command writes a `metrics.json` that records the checkpoint's SHA-256, the test split and the environment.

**HICO-DET model on HICO-DET** (about 16 minutes):

```powershell
python .\evaluate_upt_hicodet_windows.py --checkpoint .\outputs\training\hicodet\checkpoint_final.pt --output-dir .\outputs\trained-evaluation\hico-on-hico
```

**V-COCO model on V-COCO**: generate the prediction cache, then score it with the Gupta et al. evaluator:

```powershell
python .\cache_upt_vcoco_windows.py --checkpoint .\outputs\training\vcoco\checkpoint_final.pt --output-dir .\outputs\trained-evaluation\vcoco-on-vcoco
python .\evaluate_upt_vcoco_cache.py --cache .\outputs\trained-evaluation\vcoco-on-vcoco\cache.pkl --output-dir .\outputs\trained-evaluation\vcoco-on-vcoco
```

**Cross-dataset** (pass `--trained-on` so predicted classes are translated):

```powershell
python .\cache_upt_vcoco_windows.py --checkpoint .\outputs\training\hicodet\checkpoint_final.pt --output-dir .\outputs\trained-evaluation\hico-on-vcoco --trained-on hicodet
python .\evaluate_upt_vcoco_cache.py --cache .\outputs\trained-evaluation\hico-on-vcoco\cache.pkl --output-dir .\outputs\trained-evaluation\hico-on-vcoco
python .\evaluate_upt_hicodet_windows.py --checkpoint .\outputs\training\vcoco\checkpoint_final.pt --output-dir .\outputs\trained-evaluation\vcoco-on-hico --trained-on vcoco
```

**Authors' released checkpoints** (reference rows; the defaults point at `checkpoints\official`):

```powershell
python .\evaluate_upt_hicodet_windows.py --output-dir .\outputs\trained-evaluation\official-hico
python .\cache_upt_vcoco_windows.py --output-dir .\outputs\trained-evaluation\official-vcoco
python .\evaluate_upt_vcoco_cache.py --cache .\outputs\trained-evaluation\official-vcoco\cache.pkl --output-dir .\outputs\trained-evaluation\official-vcoco
python .\cache_upt_vcoco_windows.py --checkpoint .\checkpoints\official\upt-r50-hicodet.pt --output-dir .\outputs\trained-evaluation\official-hico-on-vcoco --trained-on hicodet
python .\evaluate_upt_vcoco_cache.py --cache .\outputs\trained-evaluation\official-hico-on-vcoco\cache.pkl --output-dir .\outputs\trained-evaluation\official-hico-on-vcoco
```

## 8. Reading the outputs

- **HICO-DET**: `metrics.json` holds Default Full, Rare and Non-rare mAP as fractions (multiply by 100), plus the AP of all 600 interaction classes in `per_interaction_AP`.
- **V-COCO**: `metrics.json` holds AP_role for Scenarios 1 and 2 as raw evaluator values (`scenarios_raw_AP_percent`) and with the UPT authors' 25/24 correction (`scenarios_upt_corrected_AP_percent`, the values to report), plus per-action and per-role AP.
- The V-COCO caches (`cache.pkl`) are large and are not committed. `cache-manifest.json` records the checkpoint and cache hashes.

## 9. Rebuild the figures

```powershell
pip install matplotlib
python .\make_figures.py
```

The script reads the `metrics.json` files above and writes the PNGs in `docs/figures/`.

## 10. Adaptations to disclose

This is a Windows, single-GPU adaptation. The upstream entry point uses distributed training over NCCL, which the wrappers here bypass. Training freezes the detector and optimizes the interaction head with 16 images per step accumulated over forward passes of 2 images. The UPT interaction loss clamps the positive-match normalizer to at least one so a batch with no positive matches stays finite. The V-COCO evaluator has Python 3 / NumPy compatibility changes. Details are in [PATCHES.md](PATCHES.md). Because of these changes the runs are a faithful single-GPU reproduction, not an exact copy of the authors' multi-GPU setup.

## 11. References

- UPT paper, code and model zoo: <https://github.com/fredzzhang/upt>, <https://arxiv.org/abs/2112.01838>
- HICO-DET dataset: <https://umich-ywchao-hico.github.io/>
- V-COCO annotations and evaluator: <https://github.com/s-gupta/v-coco>
- COCO 2014 download page: <https://cocodataset.org/#download>
- UPT authors' V-COCO evaluation note: <https://github.com/fredzzhang/upt/discussions/14>
