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

## B. Locally fine-tuned checkpoints

No final local training checkpoints or trained-checkpoint evaluation JSON files were present when this handoff was prepared. Fill this table only from the `metrics.json` files produced by the training-checkpoint evaluation commands in [HANDOFF.md](HANDOFF.md).

| Dataset / metric | Locally fine-tuned UPT-R50 result | Evidence file |
|---|---:|---|
| HICO-DET Default Full mAP | Not measured yet | `outputs/trained-evaluation/hicodet/metrics.json` |
| HICO-DET Default Rare mAP | Not measured yet | `outputs/trained-evaluation/hicodet/metrics.json` |
| HICO-DET Default Non-Rare mAP | Not measured yet | `outputs/trained-evaluation/hicodet/metrics.json` |
| V-COCO AP_role, Scenario 1 (UPT corrected) | Not measured yet | `outputs/trained-evaluation/vcoco/metrics.json` |
| V-COCO AP_role, Scenario 2 (UPT corrected) | Not measured yet | `outputs/trained-evaluation/vcoco/metrics.json` |

When results exist, record the final checkpoint SHA-256, the run configuration, and any deviations alongside the scores. A draft table or estimated value is not a measured result.

## References

1. F. Z. Zhang, D. Campbell, and S. Gould, “Efficient Two-Stage Detection of Human-Object Interactions with a Novel Unary-Pairwise Transformer,” CVPR 2022. Paper: <https://arxiv.org/abs/2112.01838>. Official implementation/model zoo: <https://github.com/fredzzhang/upt>.
2. UPT authors, V-COCO evaluation instructions and 25/24 correction: <https://github.com/fredzzhang/upt/discussions/14>.
3. S. Gupta and J. Malik, “Visual Semantic Role Labeling,” 2015. V-COCO code/data resources: <https://github.com/s-gupta/v-coco>.
4. HICO-DET dataset page: <https://umich-ywchao-hico.github.io/>. Dataset paper: <https://arxiv.org/abs/1702.05448>.
5. COCO 2014 images and annotations: <https://cocodataset.org/#download>.
