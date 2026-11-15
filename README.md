# UPT-R50 HOI detection: teammate guide

This project uses **one model, UPT-R50**, on two datasets in separate runs: HICO-DET and V-COCO. The working copy includes Windows training/evaluation wrappers and baseline metrics. A normal project upload excludes local datasets, model weights, caches, and upstream clones; see [HANDOFF.md](HANDOFF.md) to restore them.

Start with [HANDOFF.md](HANDOFF.md). It assumes no prior project knowledge and gives the exact run order. For results that have actually been measured, see [results.md](results.md). For local source changes, see [PATCHES.md](PATCHES.md).

## At a glance

| Item | Current setup |
|---|---|
| Model | UPT with a ResNet-50 backbone (UPT-R50) |
| Datasets | HICO-DET and V-COCO, trained/evaluated separately |
| Machine | Windows 11, NVIDIA RTX 2050 (4 GB VRAM), 16 GB system RAM |
| Python | Python 3.14.7, PyTorch 2.14.1+cu130, torchvision 0.29.1+cu130; set `UPT_PYTHON` to your interpreter |
| Dataset root | `data` |
| Official model weights | `checkpoints/official` |
| Training checkpoints | `outputs/training/<dataset>` |
| Baseline evaluations | `outputs/hicodet` and `outputs/vcoco` |

The Python path is supplied through the `UPT_PYTHON` environment variable. A teammate using another computer must configure a compatible environment and set that variable before running the PowerShell launchers. See the handoff guide.

## Important result distinction

The existing `metrics.json` files evaluate the **authors' released UPT-R50 checkpoints**. They are baseline reproduction checks, not outputs from our local fine-tuning runs. The training output folders were empty when this handoff was prepared. Do not report the baseline scores as locally trained-model scores.

## Upstream references

- UPT paper and code: <https://github.com/fredzzhang/upt>
- HICO-DET dataset: <https://umich-ywchao-hico.github.io/>
- V-COCO annotations/evaluator: <https://github.com/s-gupta/v-coco>
- COCO 2014 images/annotations used by V-COCO: <https://cocodataset.org/#download>
