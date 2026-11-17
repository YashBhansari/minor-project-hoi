# Project handoff: start here

This guide is for a teammate who is new to the project. Read it before launching training or changing files.

## 1. What the project is doing

Human-object interaction (HOI) detection asks a model to identify a person, an object, and the action connecting them—for example, a person riding a bicycle.

This project reproduces **UPT-R50**, the Unary-Pairwise Transformer with a ResNet-50 detector backbone. There is one model architecture and two independent experiments:

1. HICO-DET: train a HICO-DET UPT model, then evaluate it on the HICO-DET test split.
2. V-COCO: train a V-COCO UPT model, then evaluate it on the V-COCO test split.

The datasets have different action labels and dataset-specific weights. They are not one combined training run. UPT freezes its DETR object detector and trains the interaction head.

## 2. What is present, and what is not

Present in this project folder:

- Upstream UPT, DETR, HICO-DET, and V-COCO source checkouts under `repos/`.
- HICO-DET images/annotations and COCO 2014 images needed by V-COCO under `data/`.
- Official DETR initializations and official full UPT checkpoints under `checkpoints/official/`.
- Evaluation outputs for the **official released UPT checkpoints** under `outputs/hicodet/` and `outputs/vcoco/`.
- Windows single-process training and evaluation wrappers.

Not present at handoff time:

- Any trained checkpoint under `outputs/training/`.
- Any evaluation JSON for a locally fine-tuned checkpoint.

That means training and trained-checkpoint evaluation still need to be run. The existing baseline scores are useful for checking the evaluation setup, but do not count as results of our training runs. A values table or draft document is not evidence of a run; use the checkpoint and `metrics.json` produced by the commands below.

## 3. Machine and software

The original laptop is Windows 11 with an RTX 2050 (4 GB VRAM) and 16 GB RAM. The project was checked with Python 3.14.7, PyTorch 2.14.1+cu130, and torchvision 0.29.1+cu130. The GPU was visible to PyTorch.

Set `$env:UPT_PYTHON` to the full path of your compatible `python.exe` before running the training launchers. The original environment was checked with Python 3.14.7, PyTorch 2.14.1+cu130, and torchvision 0.29.1+cu130. The environment freeze is in `environment/windows-py314-freeze.txt`; it records installed packages but is not a one-command portable installer.

The RTX 2050 has limited VRAM. These scripts use one process and 16 images per optimizer step (the paper's batch size), accumulated over forward passes of 2 images. Run only one dataset training job at a time. Training could take many hours or days; exact runtime depends on the machine and should be measured rather than promised.

## 4. Folder map

| Path | What it contains |
|---|---|
| `repos/upt/` | Official UPT source checkout and its pinned submodules |
| `repos/v-coco/` | Gupta et al.'s V-COCO role evaluator, with documented compatibility edits |
| `data/hico_20160224_det/` | HICO-DET images |
| `data/mscoco2014/` | COCO 2014 train/val images and annotations used by V-COCO |
| `data/instances_*.json` | UPT dataset annotation files, including V-COCO splits |
| `checkpoints/official/` | Dataset-specific DETR initialization weights and released full UPT checkpoints |
| `outputs/hicodet/` | HICO evaluation of the released UPT checkpoint |
| `outputs/vcoco/` | V-COCO prediction cache and evaluation of the released UPT checkpoint |
| `outputs/training/hicodet/` | HICO training checkpoints (created by the launcher) |
| `outputs/training/vcoco/` | V-COCO training checkpoints (created by the launcher) |
| `outputs/trained-evaluation/` | Suggested location for results from locally trained checkpoints |

Do not delete the dataset or official checkpoints to make room without checking with the project owner. `data/` is excluded from version control because it is large; other machines must receive or download the data separately and match the expected folder structure.

## 5. Restore upstream source checkouts after cloning this project

The upstream clones in `repos/` are deliberately not uploaded as nested repositories. Recreate the exact source versions from PowerShell in the project folder, then apply the checked-in compatibility patches:

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

`pocket` is installed as a Python dependency; its pinned source revision is recorded in `environment/windows-py314-freeze.txt`. The local `grouphoi` clone is not imported by the training or evaluation scripts and is not required for these experiments.

The patches are important: without them the single-GPU UPT loss can divide by zero on a batch with no positive matches, and the V-COCO evaluator uses Python/NumPy interfaces removed in the current environment. Do not apply a patch twice. If a patch fails, verify the upstream revision and inspect the error instead of force-applying it.

## 6. Before training: preflight checklist

From PowerShell, set the project folder and confirm all required inputs exist. Replace the folder path if the project has moved:

```powershell
Set-Location "<path-to-your-project-clone>"

Test-Path .\data\hico_20160224_det\train2015
Test-Path .\data\hico_20160224_det\test2015
Test-Path .\data\mscoco2014\train2014
Test-Path .\data\mscoco2014\val2014
Test-Path .\checkpoints\official\detr-r50-hicodet.pth
Test-Path .\checkpoints\official\detr-r50-vcoco.pth
```

Each command should return `True`. Set `$env:UPT_PYTHON` to your environment's Python executable, then verify CUDA is available before training. If a required file is missing, stop and fix the data/setup before training. Dataset layout and verified loader counts are recorded in `data/manifest.md`.

## 7. Train the two models

Run one dataset at a time. Keep the laptop connected to power and prevent Windows from sleeping. The HICO run is a good first run because it also confirms the environment can complete training steps.

```powershell
Set-Location "<path-to-your-project-clone>"
.\train_hicodet.ps1
```

When the HICO launcher exits successfully and `outputs\training\hicodet\checkpoint_final.pt` exists, start V-COCO:

```powershell
.\train_vcoco.ps1
```

Each launcher requests 20 epochs, a batch size of 16 (gradient accumulation over passes of 2 images), and a checkpoint every 250 optimizer steps. It automatically resumes from `checkpoint_latest.pt` if interrupted. A completed run renames the latest checkpoint to `checkpoint_final.pt`. Do not run two jobs for the same dataset at once. Do not delete `checkpoint_latest.pt` during an active or interrupted run.

The trainer shows a live progress bar (step, epoch, loss, speed, estimated time left) and writes no log file. To check on it from elsewhere, use a second PowerShell window and inspect the process and checkpoint timestamp:

```powershell
Get-Process python -ErrorAction SilentlyContinue | Select-Object ProcessName,CPU,StartTime
Get-ChildItem .\outputs\training\hicodet -Force | Select-Object Name,Length,LastWriteTime
```

Use `vcoco` instead of `hicodet` in the folder path for that run. A checkpoint timestamp that advances periodically is evidence that the run is saving; it is not a performance metric. If the launcher exits with an error, keep the checkpoint and error text and diagnose before restarting. Rerunning the same launcher resumes automatically when a valid latest checkpoint exists.

## 8. Evaluate each final checkpoint

Only evaluate after `checkpoint_final.pt` exists. Use the same Python interpreter configured in the launchers.

### HICO-DET

```powershell
$python = $env:UPT_PYTHON
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'Set UPT_PYTHON to your Python executable path first.' }
& $python .\evaluate_upt_hicodet_windows.py `
  --checkpoint .\outputs\training\hicodet\checkpoint_final.pt `
  --output-dir .\outputs\trained-evaluation\hicodet
```

The resulting `metrics.json` contains Full, Rare, and Non-Rare mAP plus checkpoint hash and run environment.

### V-COCO

First generate the UPT prediction cache, then score it with the Gupta et al. evaluator:

```powershell
$python = $env:UPT_PYTHON
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'Set UPT_PYTHON to your Python executable path first.' }
& $python .\cache_upt_vcoco_windows.py `
  --checkpoint .\outputs\training\vcoco\checkpoint_final.pt `
  --output-dir .\outputs\trained-evaluation\vcoco
& $python .\evaluate_upt_vcoco_cache.py `
  --cache .\outputs\trained-evaluation\vcoco\cache.pkl `
  --output-dir .\outputs\trained-evaluation\vcoco
```

Use the corrected Scenario 1 and Scenario 2 AP_role values in `metrics.json`, and retain the raw values and the stated 25/24 correction in your notes. The cache manifest identifies the checkpoint and evaluator inputs.

## 9. How to report results

- HICO-DET: report Default Full, Rare, and Non-Rare mAP.
- V-COCO: report AP_role for Scenario 1 and Scenario 2; state whether values are raw or UPT-corrected.
- Record the exact checkpoint filename and SHA-256, test split, metric implementation, environment, and any deviations.
- Keep released-checkpoint evaluation and fine-tuned-checkpoint evaluation as separate rows/tables.
- Never infer a trained score from the released checkpoint score or type a guessed score into the results table.
- Update `results.md` only from the generated `metrics.json` files. Preserve the original evaluation outputs.

## 10. Compatibility changes and caveats

This is a Windows, single-GPU adaptation. The upstream UPT entrypoint uses distributed training and NCCL, which is not used by the local wrappers. Training freezes the detector and optimizes the interaction head, with 16 images per optimizer step accumulated over forward passes of 2 images. The local UPT loss code clamps the positive-match normalizer to avoid division by zero when a single-image batch contains no positive matches. The official V-COCO evaluator has Python 3 / NumPy compatibility changes. Details and the patch are in [PATCHES.md](PATCHES.md) and `vcoco-evaluator-python3.patch`.

These adaptations should be disclosed in a report. Do not call the experiment an exact reproduction of the authors' multi-GPU training setup.

## 11. Key references

- UPT paper/code/model zoo: <https://github.com/fredzzhang/upt>
- UPT paper: <https://arxiv.org/abs/2112.01838>
- HICO-DET dataset page: <https://umich-ywchao-hico.github.io/>
- V-COCO annotations/evaluator: <https://github.com/s-gupta/v-coco>
- COCO 2014 download page: <https://cocodataset.org/#download>
- UPT authors' V-COCO evaluation note: <https://github.com/fredzzhang/upt/discussions/14>
