# Direction 5: Failure-Focused Phase Congruency Analysis

> **Do ultrasound segmentation models fail specifically in low-contrast / ambiguous-boundary regions, and can phase congruency identify those failure-prone regions before or after prediction?**

This directory contains the minimal diagnostic experiment (Phases 0–2) to test that hypothesis on TN3K thyroid ultrasound using a standard UNet baseline.

---

## Why This Direction

Standard ultrasound segmentation work improves average Dice. This direction asks a different question first: **where does the model fail, and does phase congruency explain it?**

The target claim is:
> "X% of baseline errors occur within 5 pixels of the ground-truth boundary. Phase congruency achieves higher boundary-alignment AUC than Sobel/Canny, and high-error boundary pixels separate into two groups: low-phase ambiguous regions and high-phase ignored-structure regions."

If the data supports that claim, the direction is real and the method follows from the evidence. If it does not, the experiment tells you why — saving months of misaligned effort.

---

## Research Hypotheses

**H1 — Failure localization**
> Baseline ultrasound segmentation models fail disproportionately around weak, low-contrast, speckle-corrupted boundaries.

**H2 — Phase relevance**
> Phase-congruency response is higher near true anatomical/lesion boundaries than in random background regions, and it overlaps with boundary-error regions better than standard gradient edges.

**H3 — Phase-guided improvement** *(to be tested in Phase 3+ if H1 and H2 hold)*
> If errors concentrate in regions where phase congruency identifies structural boundaries, then phase-guided losses or post-refinement should improve boundary metrics more than global Dice.

---

## Dataset

**TN3K** — thyroid nodule ultrasound segmentation.
- HuggingFace: `haifan-gong/TN3K`
- 3,493 images total; grayscale B-mode with binary segmentation masks
- Covers benign and malignant nodules across varied contrast and boundary conditions

---

## Experiment Phases

### Phase 0 — Dataset + Baseline

Train a standard UNet on TN3K and evaluate on the test set.

**Model:** UNet with 4 encoder blocks, 512-channel bottleneck, 4 decoder blocks with skip connections. Input: 1-channel grayscale at 256×256.

**Loss:** `0.5 * BCE + 0.5 * Dice`

**Metrics:** Dice, IoU, Boundary F-score, HD95

**Expected baseline:** Dice ~0.74–0.82, HD95 ~8–18 px

---

### Phase 1 — Failure Discovery

For every test image, compute:

```
E = XOR(P, G)           # error map
FP = P=1 & G=0          # false positives
FN = P=0 & G=1          # false negatives
Band_r = dilate(boundary(G), r)   # GT boundary band at r = 3, 5, 10 px
BER_r = sum(E ∩ Band_r) / sum(E)  # Boundary Error Ratio
```

**Main question:** Is most error localized near the GT boundary?

**Decision threshold:**
- Mean BER at r=5 px **> 0.55** → boundary failure is dominant → Direction 5 is supported
- Mean BER at r=5 px **< 0.40** → errors are diffuse → hypothesis is weaker

---

### Phase 2 — Phase Relevance

Compute phase-congruency (PC) maps using Kovesi's log-Gabor method (implemented via scipy FFT — no external library required) and compare against Sobel and Canny baselines.

**Metrics:**

| Metric | Question answered |
|--------|------------------|
| `AUC(PC, GT boundary band)` | Does PC detect where boundaries are? |
| `AUC(Sobel/Canny, GT boundary band)` | Baseline comparison |
| `AUC(PC within Band5, errors within Band5)` | Does PC predict where errors occur? |
| Mean PC in: boundary band / inside object / background | Is PC concentrated at boundaries? |
| Error rate by PC quartile (Q1=low, Q4=high) within Band5 | Do errors cluster in strong-phase or weak-phase regions? |

**Quartile interpretation:**

| Pattern | Meaning | Follow-up method |
|---------|---------|-----------------|
| High error in Q1 (low PC) | Ambiguous boundaries — no stable structure | Uncertainty-aware training |
| High error in Q4 (high PC) | Model ignores visible phase cues | Phase-guided boundary loss |
| High in both Q1 and Q4 | Two distinct failure modes | Hybrid method |

---

## Four-Region Failure Taxonomy

Near GT boundaries, each pixel falls into one of four types:

| Type | Error? | Phase high? | Meaning |
|------|--------|------------|---------|
| 1 | No | Yes | Model follows visible structure correctly |
| 2 | Yes | Yes | Model ignores useful phase boundary cue |
| 3 | Yes | No | True ambiguous / fuzzy boundary |
| 4 | No | No | Model succeeds despite weak phase |

**Type 2** motivates phase-guided correction. **Type 3** motivates uncertainty-aware training. Both are publishable — they lead to different methods and both are grounded in interpretable evidence.

---

## Visualizations

Each test image produces a 2×4 diagnostic panel:

```
Row 1: [Ultrasound] [GT overlay] [Prediction overlay] [Error map: FP=red, FN=blue]
Row 2: [PC map]     [PC edges]   [Error over PC]      [Boundary band + error]
```

Three failure galleries are generated:
- **Gallery A** — Phase highlights the missed boundary (high BER + high PC in error region)
- **Gallery B** — Failures occur at genuinely ambiguous low-phase boundaries
- **Gallery C** — Phase is misleading (highlights speckle or non-nodule interfaces)

Gallery C is important: it shows the method's limits and makes the analysis more credible.

---

## Metrics Beyond Dice

Region metrics: Dice, IoU, Precision, Recall

Boundary metrics: Boundary F-score (BF), HD95, Average Surface Distance

Failure-specific metrics:
```
Boundary Error Rate        = errors inside GT boundary band / pixels in band
Low-Phase Boundary Error   = errors in low-PC boundary pixels / low-PC boundary pixels
High-Phase Miss Rate       = FN in high-PC boundary regions / high-PC boundary pixels
```

---

## Setup

```bash
pip install torch torchvision datasets huggingface-hub
pip install scikit-image scikit-learn scipy matplotlib pandas numpy
pip install SimpleITK          # HD95 computation
```

No additional library is needed for phase congruency — it is implemented via `numpy.fft`.

---

## Running the Experiment

```bash
cd direction_5

# Full pipeline (all phases)
python run_experiment.py --config config.yaml

# Individual phases
python run_experiment.py --config config.yaml --phases 0_train
python run_experiment.py --config config.yaml --phases 0_eval
python run_experiment.py --config config.yaml --phases 1_failure
python run_experiment.py --config config.yaml --phases 2_phase
python run_experiment.py --config config.yaml --phases viz
```

Each module is also runnable standalone (`python train.py`, `python failure_analysis.py`, etc.) for debugging.

---

## Outputs

```
outputs/
├── checkpoints/        best_model.pth, last_model.pth
├── predictions/        {id}_pred.png, {id}_error.npy
├── metrics/            train_log.csv, test_metrics.csv,
│                       failure_metrics.csv, phase_metrics.csv
├── phase_maps/         {id}_pc.npy, {id}_sobel.npy, {id}_canny.npy
└── figures/            panel_{id}.png, gallery_{A,B,C}.png,
                        ber_violin.png, auc_comparison.png,
                        quartile_error_rates.png
```

---

## Decision Tree After Phase 1 + 2

**Scenario 1: Boundary errors dominant + phase aligns with GT boundaries**
→ Phase-guided boundary supervision (`Loss = Dice + CE + λ * PhaseBoundaryLoss`)

**Scenario 2: Boundary errors dominant + phase is weak in error regions**
→ Phase-derived uncertainty modeling (`loss_weight(x) = f(PC(x))`)

**Scenario 3: Boundary errors not dominant**
→ Direction 5 is weaker; investigate class imbalance, domain shift, or shadowing artifacts instead

**Scenario 4: Phase does not align with GT boundaries and does not explain errors**
→ Stop or reframe; check phase parameters, annotation sharpness, or target structure type

---

## Suggested Paper Titles

- *Failure-Aware Ultrasound Segmentation via Phase-Congruency-Based Structural Error Analysis*
- *Do Ultrasound Segmentation Models Fail Where Structure Disappears? A Phase-Congruency Analysis of Boundary Errors*
