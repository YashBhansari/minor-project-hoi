# Results and provenance

This file separates results measured with the authors' released checkpoints from results that must be measured with locally fine-tuned checkpoints.

## A. Evaluation of the authors' released UPT-R50 checkpoints

These values are in `outputs/hicodet/metrics.json` and `outputs/vcoco/metrics.json`. The model weights are under `checkpoints/official/upt-r50-*.pt`. They are baseline evaluation results, **not** results from the local training scripts.

| Dataset / metric | UPT paper reference | Local evaluation of released UPT checkpoint | Difference |
|---|---:|---:|---:|
| HICO-DET Default Full mAP | 31.66 | 31.7825 | +0.1225 |
| HICO-DET Default Rare mAP | 25.94 | 26.2417 | +0.3017 |
| HICO-DET Default Non-Rare mAP | 33.36 | 33.4376 | +0.0776 |
| V-COCO AP_role, Scenario 1 (UPT corrected) | 59.0 | 58.9583 | -0.0417 |
| V-COCO AP_role, Scenario 2 (UPT corrected) | 64.5 | 64.4792 | -0.0208 |

HICO scores are stored as fractions in JSON and shown here as percentages. V-COCO's evaluator returned raw AP_role values of 56.60 and 61.90; the UPT note recommends multiplying by 25/24, yielding 58.9583 and 64.4792. Keep raw and corrected values distinct.

The local values are close to the paper references. This supports that the current local data and evaluation paths can reproduce the authors' released-checkpoint behavior. It does not establish the performance of a newly fine-tuned checkpoint.

## B. Locally trained V-COCO model

Trained here with `train_vcoco.ps1`: 20 epochs, 6,200 steps, batch 16 (8 passes of 2 images, gradients accumulated on one GPU), AdamW lr 1e-4, weight decay 1e-4, lr ÷10 at epoch 10, clip 0.1, seed 66, frozen official `detr-r50-vcoco.pth`. Final checkpoint SHA-256 `61dbd4a0e7c47c629b1ded5f913c6acb7374b4d09eb27a95eedb3bc570acdcda`.

| V-COCO metric | Paper | Authors' released model (rerun) | **Locally trained** |
|---|---:|---:|---:|
| AP_role, Scenario 1 (corrected) | 59.0 | 58.94 | **59.03** |
| AP_role, Scenario 2 (corrected) | 64.5 | 64.46 | **64.60** |

Raw evaluator values for the trained model are 56.67 and 62.02. Evidence: `outputs/trained-evaluation/vcoco-on-vcoco/metrics.json`.

**HICO-DET was not trained locally.** Training on the available laptop GPU needs about 18 hours, so the HICO-DET model in section C is the authors' released checkpoint (SHA-256 `12f7f84c7c8d5bcc38f53470f57c5cc30ff2f025726e8bcc27addeb0578a258c`).

## C. Cross-dataset evaluation (not reported in the paper)

A model trained on one dataset is scored on the other by translating its predicted classes. Only classes with a counterpart are translated (identical verb names plus 14 hand-checked pairs listed in `cross_dataset.py`); other predictions are dropped, and target classes with no counterpart score zero.

| Model | Tested on HICO-DET (Default Full mAP) | Tested on V-COCO (AP_role S1 / S2, corrected) |
|---|---:|---:|
| HICO-DET model (authors' weights) | 31.72 (rare 26.09, non-rare 33.41) | 44.35 / 48.30 |
| V-COCO model (locally trained) | 7.03 (rare 1.57, non-rare 8.67) | 59.03 / 64.60 |

Evidence: `outputs/trained-evaluation/{hico-on-hico,vcoco-on-hico,hico-on-vcoco,vcoco-on-vcoco}/metrics.json`. Diagonal cells are same-dataset results; off-diagonal cells are the cross-dataset results.

Coverage caveats, which explain most of the cross-dataset drops:

- **V-COCO model on HICO-DET:** only 147 of 600 interaction classes are reachable (21 of 117 verbs, restricted to objects the V-COCO model can pair them with). Over those 147 classes the mAP is **28.71**; the other 453 score zero by construction. Verified: all 147 reachable classes have non-zero AP and none outside them do. 28.71 is over a different class mix than the paper's 31.66, so it is not a like-for-like comparison.
- **HICO-DET model on V-COCO:** 23 of 24 actions have a counterpart; `eat instr` has none and scores 0.13. `point instr` scores 0.00 for every model and is excluded by the 25/24 correction.
- The pairings are judgement calls; the loosest are `look obj`/`watch` and `hit instr`/`swing`.

Section A was measured earlier (31.78 HICO-DET full mAP; raw V-COCO 56.60 / 61.90). Re-running the same released checkpoints for section C gave 31.72 and 56.58 / 61.88. The small differences (under 0.1) were not investigated; they are most likely run-to-run GPU numerical differences, but that is unconfirmed.

## References

1. F. Z. Zhang, D. Campbell, and S. Gould, “Efficient Two-Stage Detection of Human-Object Interactions with a Novel Unary-Pairwise Transformer,” CVPR 2022. Paper: <https://arxiv.org/abs/2112.01838>. Official implementation/model zoo: <https://github.com/fredzzhang/upt>.
2. UPT authors, V-COCO evaluation instructions and 25/24 correction: <https://github.com/fredzzhang/upt/discussions/14>.
3. S. Gupta and J. Malik, “Visual Semantic Role Labeling,” 2015. V-COCO code/data resources: <https://github.com/s-gupta/v-coco>.
4. HICO-DET dataset page: <https://umich-ywchao-hico.github.io/>. Dataset paper: <https://arxiv.org/abs/1702.05448>.
5. COCO 2014 images and annotations: <https://cocodataset.org/#download>.
