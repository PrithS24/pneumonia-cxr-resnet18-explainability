# Pediatric Pneumonia Classification on Chest X-rays: ResNet-18, Explainability and Tuning

An end-to-end study of a chest X-ray classifier (normal vs. pneumonia) on PneumoniaMNIST. The notebook goes beyond training a model: it checks the data for leakage, evaluates calibration and error types, tests whether the explanation maps can be trusted, and examines why hyperparameter tuning on a validation set can fail under distribution shift.

All experiments run in a single notebook on one NVIDIA T4 GPU (Kaggle). The notebook is saved with its outputs, so all figures and tables can be viewed directly on GitHub without re-running it.

## Highlights

- **Data checks before training.** MD5 hashing found 18 duplicate images within the training set and 8 validation images that were also in training. These were removed to prevent train-to-validation leakage.
- **Evaluation beyond accuracy.** AUROC with bootstrap confidence intervals, AUPRC, sensitivity, specificity, calibration (Brier score, ECE) and an analysis of high-confidence errors.
- **Explanations that are checked, not just shown.** Grad-CAM is tested with a model-randomization sanity check and cross-checked against occlusion sensitivity, a gradient-free method.
- **An honest tuning result.** Optuna improved validation metrics in every seed but made test performance worse in every seed. The notebook shows why: the validation set follows the training distribution, while the test set is shifted.
- **Speed optimization.** Mixed precision, `channels_last` memory format and input resolution are benchmarked for throughput, memory and accuracy.

## Dataset

[PneumoniaMNIST](https://medmnist.com/) from MedMNIST v2, at 224 × 224 resolution. It is derived from the pediatric chest X-ray dataset of Kermany et al. (2018) and released under CC BY 4.0. The notebook downloads it automatically through the `medmnist` package, so no manual download is needed.

| Split | Images | Normal | Pneumonia | Pneumonia share | After deduplication |
|---|---|---|---|---|---|
| Train | 4,708 | 1,214 | 3,494 | 74.2% | 4,690 |
| Validation | 524 | 135 | 389 | 74.2% | 516 |
| Test | 624 | 234 | 390 | 62.5% | 624 (unchanged) |

The test set has a lower pneumonia share than training, so it exhibits prior (label) shift. The official test set was kept intact for comparability.

## Method

| Component | Choice |
|---|---|
| Model | ResNet-18, ImageNet-pretrained, all layers fine-tuned, single-logit output |
| Input | Grayscale replicated to 3 channels, ImageNet normalization |
| Augmentation | Small rotation, translation, scaling, brightness and contrast changes. No horizontal flips, because chest X-rays are not left-right symmetric |
| Loss | Binary cross-entropy with logits, `pos_weight` = N_normal / N_pneumonia ≈ 0.35 for class imbalance |
| Optimizer | AdamW (lr 1e-4, weight decay 1e-4), cosine learning-rate schedule, batch size 64 |
| Training | Mixed precision, up to 12 epochs, early stopping (patience 4) and model selection on validation loss |
| Threshold | 0.5. A Youden-optimal threshold chosen on validation is reported as a secondary analysis |

## Results

### Baseline classifier (test set, threshold 0.5)

| AUROC (95% CI) | AUPRC | Accuracy | Sensitivity | Specificity | FP | FN |
|---|---|---|---|---|---|---|
| 0.9926 (0.9873 to 0.9971) | 0.9954 | 0.952 | 0.990 | 0.889 | 26 | 4 |

- Errors are mostly false positives on healthy cases. Validation specificity was much higher (0.978), which points to differences between the validation and test images rather than a general bias toward pneumonia.
- Aggregate calibration is reasonable (ECE 0.037, Brier 0.038), but 16 of the 30 errors were made with high confidence (p > 0.9 or p < 0.1).
- The validation-selected Youden threshold (0.415) added 3 false positives on test without catching any extra pneumonia cases.

### Explainability

`explain_prediction()` produces a Grad-CAM heatmap for any image. Two checks test whether these maps are trustworthy:

| Check | Result |
|---|---|
| Model-randomization test (32 images) | Mean correlation between trained-model and random-weight maps r = −0.39, so the maps depend on the learned weights |
| Grad-CAM vs. occlusion agreement (412 images predicted as pneumonia) | Median per-image Spearman ρ ≈ 0.55 (moderate agreement) |
| Border attention, false vs. true positives | Grad-CAM: 0.41 vs. 0.37 (p = 0.046). Occlusion: 0.21 vs. 0.41 (p < 0.001) |

The two methods disagree on whether false positives rely on image borders, so no shortcut-learning claim is made. Occlusion instead places false-positive evidence in a few central regions near the mediastinum.

### Hyperparameter tuning

Optuna (TPE sampler, median pruning, 20 trials of 8 epochs) minimized validation loss over learning rate, weight decay, batch size, augmentation and dropout. Baseline and tuned configurations were each trained with three seeds.

| | Baseline | Tuned |
|---|---|---|
| Validation loss | 0.052 ± 0.004 | **0.024 ± 0.003** |
| Test AUROC | **0.9933 ± 0.0006** | 0.9862 ± 0.0045 |
| Test specificity | **0.889 ± 0.000** | 0.771 ± 0.017 |
| Test false positives (mean) | **26** | 54 |

The tuned configuration (higher learning rate, lower weight decay, no augmentation) fit the validation set better but generalized worse to the shifted test set. Cross-validation would not fix this, since every fold comes from the training distribution. Reliable tuning would need validation data from the target distribution.

A validation-independent alternative, averaging the predictions of the three baseline seeds, gave a small gain: test AUROC 0.9945 (95% CI 0.9907 to 0.9977), 24 false positives and the lowest Brier score (0.035).

### Speed (T4 GPU)

| Setting | Train img/s | Speed-up | Peak memory (MB) | Test AUROC (3 seeds) | Train time (min) |
|---|---|---|---|---|---|
| FP32, 224 px | 332 | 1.00× | 4,152 | 0.9929 ± 0.0018 | 2.98 |
| AMP, 224 px | 695 | 2.10× | 1,306 | 0.9933 ± 0.0006 | 1.34 |
| AMP + channels_last | 921 | 2.78× | 1,797 | not trained | n/a |
| AMP, 128 px | 1,932 | 5.83× | 1,270 | 0.9896 ± 0.0039 | 0.69 |

Mixed precision halved training time and cut peak memory by about 69% with no measurable loss in AUROC. Lowering the resolution to 128 px was faster again but reduced specificity and made results less stable across seeds.

## Results workbook

`results_workbook.xlsx` contains every result table produced by the notebook, one sheet per output file, with a README sheet describing each one. It includes class balance and duplicate checks, the training history, baseline metrics, per-image Grad-CAM and occlusion results, all Optuna trials, per-seed comparisons, the seed ensemble and the speed benchmarks.

## How to run

1. Open the notebook on Kaggle (or any machine with a CUDA GPU) and enable **GPU T4** and **Internet**.
2. Run all cells from top to bottom. The first cell installs any missing packages.
3. For a quick pipeline check, set `FAST_DEV_RUN = True` in the configuration cell (about 10 minutes). The full run takes about 60 to 90 minutes on a T4.

Figures and result tables are saved to `outputs/` (`/kaggle/working/outputs` on Kaggle). GPU training is not fully deterministic, so a re-run gives slightly different numbers.

To run locally:

```bash
pip install -r requirements.txt
jupyter notebook pneumoniamnist_resnet18_explainability.ipynb
```

## Repository structure

```
├── pneumoniamnist_resnet18_explainability.ipynb   # full pipeline, saved with all outputs and figures
├── results_workbook.xlsx                           # all result tables, one sheet per output file
├── requirements.txt
└── README.md
```

## Limitations

- Single dataset from one pediatric source. Performance on other hospitals, scanners or age groups is unknown.
- No demographic metadata, so subgroup (fairness) analysis was not possible.
- Grad-CAM is coarse (7 × 7 before upsampling), occlusion depends on patch size, and without lung masks the border metric is only a proxy for attention outside the lungs.
- Three seeds are enough to see consistent effects, not to estimate small differences precisely.
- Several confident false positives show no obvious abnormality to a non-expert. They may reflect model error or label noise, which would need radiologist review.

## References

1. D. S. Kermany et al., "Identifying medical diagnoses and treatable diseases by image-based deep learning," *Cell*, 2018. https://doi.org/10.1016/j.cell.2018.02.010
2. J. Yang et al., "MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification," *Scientific Data*, 2023. https://arxiv.org/abs/2110.14795
3. K. He et al., "Deep residual learning for image recognition," *CVPR*, 2016. https://arxiv.org/abs/1512.03385
4. R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization," *ICCV*, 2017. https://arxiv.org/abs/1610.02391
5. J. Adebayo et al., "Sanity checks for saliency maps," *NeurIPS*, 2018. https://arxiv.org/abs/1810.03292
6. M. D. Zeiler and R. Fergus, "Visualizing and understanding convolutional networks," *ECCV*, 2014. https://arxiv.org/abs/1311.2901
7. J. R. Zech et al., "Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs," *PLOS Medicine*, 2018. https://doi.org/10.1371/journal.pmed.1002683
8. T. Akiba et al., "Optuna: A next-generation hyperparameter optimization framework," *KDD*, 2019. https://arxiv.org/abs/1907.10902
9. P. Micikevicius et al., "Mixed precision training," *ICLR*, 2018. https://arxiv.org/abs/1710.03740
10. B. Lakshminarayanan et al., "Simple and scalable predictive uncertainty estimation using deep ensembles," *NeurIPS*, 2017. https://arxiv.org/abs/1612.01474

## Author

Pritha Saha
