# F-path design: Ghostbuster-style LR ensemble scorer

Goal: replace the 40-pt stat-cap architecture with a single logistic-regression
ensemble over the continuous features we already compute, so that signals
like Binoculars / MATTR / curvature / transition density (all previously
"parked disabled" due to cap-saturation) can contribute jointly without
penalizing human text via correlated flagging.

Inspired by Ghostbuster (NAACL 2024, arxiv 2305.15047): multi-scale
ngram probabilities + combining via logistic regression.

## Scope

- Adds a new **scoring mode** (`--lr`) to `detect_cn`. Initial plan: keep
  `rule+stat` as default, expose `--lr` as opt-in. Promote to default in F-8
  only if it dominates on HC3 and heroes.
- No humanize-side changes in F-path. Humanize continues to optimize for
  detect_cn rule+stat via existing pipeline.

## Feature vector (18 continuous features)

Pull directly from `ngram_model.analyze_text(text)`:

1. `perplexity` — overall char-level ngram ppl (scale ~50-500)
2. `burstiness` — CV of windowed perplexity
3. `entropy_cv` — per-paragraph entropy CV
4. `diveye.skew` — DivEye surprisal skewness
5. `diveye.excess_kurt` — DivEye surprisal kurtosis
6. `diveye.spectral_flatness` — auxiliary DivEye metric
7. `diveye.autocorr_lag1` — auxiliary
8. `gltr.top10_frac` — GLTR top-10 bucket proportion
9. `gltr.mid_rank_frac` — GLTR mid-rank proportion
10. `sent_len.cv` — sentence-length coefficient of variation
11. `sent_len.short_frac` — fraction of sentences < 10 CN chars
12. `sent_len.long_frac` — fraction > 30 CN chars
13. `sent_len.equal_mid_frac` — fraction in 15-25 band
14. `punct.comma_density` — commas per 100 non-ws chars
15. `punct.punct_density` — total punctuation density
16. `trans.density` — transition-word density (parked indicator)
17. `curv.curvature_mean` — DetectGPT-lite curvature (parked indicator)
18. `bino.mean_lp_diff` — Binoculars dual-ngram log-prob delta (parked, gated on secondary file)

Char-count `char_count` feeds gating (skip if < 100) but not the LR.

Additional candidate (not in v1): `char_mattr` from E-8 (d=0.70). Currently
the indicator is disabled, but metric is still in analyze_text output. Add
as feature 19 if v1 baseline performs well and we want more capacity.

## Output scale mapping

LR gives `p_ai ∈ [0, 1]` = probability text is AI-written.

Map to `score ∈ [0, 100]`:
- `score = round(100 * p_ai)` as a first pass
- Calibration check: on HC3 300+300 holdout, verify that humans cluster
  near 10-25 and AI near 60-85 (matching current rule+stat distribution)
- If distribution is too compressed (everything in 40-60), apply linear
  stretch: `score = round(100 * clip((p_ai - lo) / (hi - lo), 0, 1))` where
  lo/hi are 10th/90th percentiles of AI sample probs

Badge thresholds (same as existing):
- 0-30: LOW 🟢
- 30-60: MEDIUM 🟡
- 60-80: HIGH 🟠
- 80-100: VERY HIGH 🔴

## Training set

- Source: HC3-Chinese all.jsonl (`/data/hc3_chinese_all.jsonl`)
- Balanced 300+300 human vs ChatGPT, both filtered to ≥ 100 Chinese chars
- Random seed 42, shuffle before split
- 80/20 train/holdout: 240+240 train, 60+60 holdout
- Holdout used for F-6 threshold tuning and final regression

Notes on data leakage: we already trained `ngram_freq_cn_human.json` on
80% of HC3 human answers (B-path). That training set overlaps with this
LR training set. For an honest Binoculars feature, we'd need to use the
existing `ngram_freq_cn_human_holdout.json` line indices to exclude
training samples. Handle in F-3 feature extraction.

## Model

Logistic regression, L2 regularized (`C=1.0` default).

- No hidden layer (interpretability matters; `coef_` → per-feature weight shown in diagnostics)
- Features standardized to mean 0 / std 1 before fitting (scaler `mean_` and `scale_` saved alongside coef)
- Implemented with sklearn if available (`pip install scikit-learn`); fall back to pure-numpy LR via gradient descent on standardized features
- Fallback prefered for zero-deps constraint — sklearn only if readily available in user env

## File artifacts

- `scripts/lr_coef_cn.json` — coefficients, bias, feature order, scaler stats
  ```json
  {
    "version": "3.4.0",
    "features": ["perplexity", "burstiness", ...],
    "mean": [259.3, 0.21, ...],
    "scale": [72.1, 0.08, ...],
    "coef": [0.31, -0.18, ...],
    "intercept": -0.45,
    "trained_on": "hc3_chinese_all.jsonl",
    "n_train": 480,
    "holdout_accuracy": 0.82
  }
  ```
- `scripts/train_lr_scorer.py` — training script
- New function in `ngram_model.py`: `compute_lr_score(text, coef_path=None)`
- New function in `detect_cn.py`: `calculate_score_lr(issues, metrics, text)`

## Acceptance gates

- F-6 regression: HC3 100-sample correct ≥ 75%, gap ≥ 14.8
- F-7 hero: academic aggressive 24 ± 3, academic normal 32 ± 3, general 36 ± 3,
  xhs 35 ± 3 under `--lr` mode (small tolerance: LR threshold calibration is
  always slightly different from rule+stat, so exact match not required)
- F-8 promotion criterion: LR mode gap > 14.8 AND worst-case HC3 sample
  delta >= current rule+stat worst case (-8)

If gates fail, document which feature combinations cause what and treat as
a v3.5 candidate. Don't promote a worse architecture.
