# Final Results

All results below are from the locked, untouched test set. Model selection, fusion weighting, and experimental decisions were made before test evaluation.

## Multimodal Performance

| Model | AUROC | PR-AUC |
|---|---:|---:|
| Image-only ResNet-18 | 0.8584 | 0.4026 |
| Text-only TF-IDF + Logistic Regression | 0.6676 | 0.1871 |
| **Multimodal Fusion (80% Image / 20% Text)** | **0.8755** | **0.4110** |

Relative to the image-only model, multimodal fusion improved:

- **AUROC:** +0.0171
- **PR-AUC:** +0.0084

## Pairwise Ranking Analysis

To investigate the AUROC improvement, positive-negative report pairs were compared before and after fusion.

| Pairwise Outcome | Count |
|---|---:|
| Rankings repaired by fusion | 513 |
| Rankings broken by fusion | 97 |
| Stayed correctly ranked | 20,752 |
| Stayed incorrectly ranked | 2,926 |
| **Net ranking improvement** | **+416** |

Across 24,288 positive-negative pairs, fusion produced a net gain of **416 correctly ordered pairs**, corresponding to the observed AUROC improvement of approximately **+0.0171**.

## Probability Quality

| Model | Brier Score | Log Loss |
|---|---:|---:|
| Image-only | 0.0660 | 0.2236 |
| Text-only | 0.0765 | 0.2811 |
| **Multimodal Fusion** | **0.0641** | **0.2201** |

Fusion therefore produced a modest improvement in probability quality relative to the image-only model.

## Failure-Mode Analysis

Fusion did not improve every individual prediction.

- Individual probability error improved in **150 / 554 reports (27.1%)**
- Individual probability error worsened in **404 / 554 reports (72.9%)**

This is not inconsistent with the AUROC improvement. AUROC measures **ranking quality across positive-negative pairs**, whereas individual absolute probability error measures how close each predicted probability is to its binary target.

High-disagreement cases showed both behaviors:

- Low-risk clinical indication signals corrected some image-model overpredictions.
- Low text probabilities also reduced confidence in several correctly identified positive image cases.

Overall, pre-imaging clinical context provided **complementary ranking information**, but its contribution was heterogeneous rather than uniformly beneficial.