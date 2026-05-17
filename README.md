# Phase Congruency for Ultrasound Segmentation

A research project exploring how phase congruency — a contrast/illumination-invariant structural signal — can address core failure modes in ultrasound segmentation.

---

## The Core Insight

Ultrasound segmentation fails because of speckle noise, low-contrast boundaries, and fuzzy edges. Phase congruency is well-matched to these failures: it gives illumination-invariant boundary localization, robustness to contrast variation, and structural (geometric, not intensity-based) information. The research question is not "can phase congruency improve Dice?" but rather:

> **Where and why does the model fail, and can phase congruency explain and guide correction of those failures?**

---

## Research Directions

### Direction 1 — Boundary-Aware Supervision

Use phase-congruency edge maps as an explicit learning signal, not just a feature.

**Approach A — Auxiliary loss:**
```
Loss = Dice + λ * BoundaryLoss(pred_mask, phase_edges)
```
Force predicted segmentation boundaries to align with phase-detected structural edges.

**Approach B — Edge consistency constraint:**
Penalise predicted edges that disagree with stable phase structure.

**Why it is strong:** Moves from "feature tweak" to "learning signal". Directly targets the dominant segmentation failure mode (boundary misalignment). Interpretable and principled.

---

### Direction 2 — Uncertainty Modeling

Phase congruency highlights structurally stable regions. Low phase response = genuine structural ambiguity.

**Idea:** Use phase strength as a spatial confidence map.
```
Loss = w(x) * segmentation_loss   where w(x) = phase_strength(x)
```
Or explicitly model uncertainty as inverse phase consistency.

**Why it is strong:** This is a paradigm shift — the model learns where it should be uncertain. Publishable even without large accuracy gains, because it addresses calibration and reliability, not just Dice.

---

### Direction 3 — Curriculum Learning

Train progressively based on structural complexity derived from the phase map.

1. Compute phase-congruency map per image.
2. Rank pixels or images by phase-derived "difficulty" (high phase = clear structure = easy; low phase = ambiguous = hard).
3. Train: easy (clear boundaries) first, then hard (noisy/ambiguous) regions.

**Why it is strong:** Reduces optimisation instability. Aligns training order with perceptual complexity. Underexplored in medical imaging, especially ultrasound.

---

### Direction 4 — Multi-Branch Architecture

Instead of injecting phase into attention (where it gets diluted), use a parallel stream.

```
Stream 1: CNN/ViT(raw_image)
Stream 2: CNN(phase_map)
Output:   Fusion(stream1_features, stream2_features)
```

Fusion strategies to compare: early (input-level), mid (feature-level alignment), late (prediction-level ensemble).

**Why it is better than attention injection:** Preserves the phase signal explicitly. Avoids attention dilution in large transformer models. Each stream is independently analysable.

---

### Direction 5 — Failure-Focused Analysis (Active: see `direction_5/`)

The strongest framing. Prove the failure mode first, then show phase congruency diagnoses and targets it.

**Step 1:** Find where the baseline model fails: weak edges, small structures, noisy/low-contrast regions.

**Step 2:** Test whether phase congruency highlights those regions — before or after prediction.

**Step 3:** Use that alignment to target corrections: adaptive loss, uncertainty weighting, or boundary refinement.

**Why it is strong:** Instead of claiming a uniform average improvement, this produces the statement:
> *"Our method improves performance specifically in low-contrast boundary regions by X%"*
That is a fundamentally stronger claim than average Dice gain.

---

## Paper Framing

**Weak framing (avoid):**
> "We introduce phase-congruent attention for ultrasound segmentation"

**Strong framing:**
> "We address boundary ambiguity in ultrasound segmentation using phase-based structural priors"

**Contribution structure:**
1. Phase-guided boundary supervision
2. Uncertainty estimation via phase consistency
3. Robust segmentation under low contrast and speckle

This positions the work as principled, problem-driven, and motivated by a documented failure mode — not as an architecture tweak hoping for incremental gains.

---

## What Strong Papers Do

- Tie the idea to the problem tightly, not loosely
- Inject signal at the **decision level** (loss, output), not just the feature level
- Show **where** the method matters, not only average metric improvements
- Identify the failure mode first, then design the fix

---

## Repository Structure

```
phase-congruency-ultrasound/
├── README.md                   # This file
└── direction_5/                # Phase 0–2: failure discovery + phase relevance
    ├── README.md               # Direction 5 experiment guide
    ├── config.yaml
    ├── dataset.py
    ├── model.py
    ├── train.py
    ├── evaluate.py
    ├── failure_analysis.py
    ├── phase_analysis.py
    ├── visualize.py
    └── run_experiment.py
```
