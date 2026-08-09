# Clustering Algorithms for Regression-Like Modeling
### Applications in Market Segmentation & Predictive Maintenance

**Master's thesis** — Advanced Master in Artificial Intelligence & Business Transformation (AIBT), ISAE-SUPAERO
Completed as a 1-year apprenticeship at **Airbus** (Oct 2023 – Sep 2024) · Defended 19/09/2024
Tutors: Vlad Marin (Airbus) · Rob Vingerhoeds (ISAE-SUPAERO)

📄 **[Read the full report (PDF, 51 pages)](Master-Thesis_Report_Abbes.pdf)**

---

## Overview

Two independent, industry-embedded studies exploring the same idea from opposite ends: using **unsupervised learning to solve problems that are usually framed as supervised regression**, on real Airbus data, under real data-availability constraints.

| | Study 1 — Digital Maturity | Study 2 — Predictive Maintenance |
|---|---|---|
| **Goal** | Score airlines on digital maturity from platform usage data | Forecast a component's degradation index from operating conditions |
| **Method** | Kernel PCA + Fuzzy C-Means, with an expert-informed scoring layer | Two-phase (unsupervised → supervised) Self-Organizing Map |
| **Data** | 159 airlines, Skywise usage + fleet features | 579 aircraft, 27 features, only 8.47% labeled |
| **Headline result** | Silhouette score of **0.804**; results cross-checked against two independent published industry reports | Beat XGBoost, Random Forest, and Bayesian Symbolic Regression on test MAE by a wide margin |
| **Business outcome** | Dashboard deployed live on Skywise, used by the marketing team | Identified 3 distinct, physically-interpretable degradation root causes |

---

## Study 1 — Estimating Airline Digital Maturity

**Context.** Airbus's understanding of the digital-solutions market relied on third-party consulting reports and informal customer feedback — not sustainable, and blind to airlines outside existing surveys. The goal was a generic, internally-owned, data-driven method to rank airlines by digital maturity, using Skywise (Airbus's big-data platform, 47,000+ monthly active users across 150+ airlines) usage data instead of external surveys.

**Approach.**
1. **Preprocessing**: min-max scaling, then expert-assigned feature weights (1–5) so the marketing team's domain knowledge of what actually signals digital maturity shapes the clustering, not just raw variance.
2. **Dimensionality reduction**: benchmarked PCA against Kernel PCA across component counts on both conserved variance and downstream silhouette score — Kernel PCA (2 components, 86.12% variance conserved) won on the variance/silhouette trade-off, reflecting the nonlinear structure of the feature set.
3. **Clustering**: Fuzzy C-Means over hard clustering (k-means) — because the deliverable wasn't a label, it was a continuous *maturity score*, and FCM's membership matrix (how strongly each airline belongs to each cluster) is exactly the right primitive for that.
4. **Scoring**: clusters are ranked by an expert-defined, feature-weighted score (co-designed with a marketing manager), then each airline's fuzzy memberships are combined into a single Digital Maturity Index and rescaled for a smooth, interpretable distribution.

**Results.** The clustering converged with a silhouette score of **0.804** (0.7+ is generally considered strong). Rather than stopping at the internal metric, the top-ranked airlines were cross-checked against two independent, external sources — the Lufthansa Innovation Hub's *Airline Digital Index* and the Remarkable Group's *2024 Digital Maturity Report* — and the rankings lined up with both. The model was also honest about its own limits: it separates the clear top and bottom performers well but is less confident in the middle band, a nuance confirmed directly by the marketing experts reviewing the output.

**Impact.** Shipped as a live two-tab dashboard inside Skywise — airline-level profiles and maturity scores for the marketing team, plus a methodology/monitoring tab (silhouette score, conserved variance, last-update date, feature importances) so the model's health is auditable by non-data-scientists. It's now used to prioritize upsell targeting across Skywise's paid tiers.

---

## Study 2 — Predicting Component Degradation for Predictive Maintenance

**Context.** Built with Airbus's Zero AOG team (Toulouse), whose mission is minimizing aircraft downtime. The goal: predict a continuous degradation index for a non-critical ("non-No-Go") component from its operating-condition history, to move maintenance from reactive to proactive — and, longer-term, to feed root-cause insight back into the component's design. (Equipment and dataset are anonymized in the report per Airbus confidentiality requirements.)

**The real constraint: almost no labels.** Of 579 samples, only 49 (8.47%) had a measured degradation index. This ruled out a standard supervised pipeline and motivated the choice of method.

**Approach.** Self-Organizing Maps, extending the two-phase framework of Riese & Keller (2018): an unsupervised SOM first learns the topology of the *input* space from all available data (labeled or not), then a second, supervised SOM of the same size learns to map each node's Best Matching Unit to the *target* value using only the labeled points. Two variants were run and compared:
- **Fully supervised** — trained only on the 39 labeled training points. The standard sizing rule of thumb suggests a small 6×6 map, which under-discretizes a continuous 0–1 target (only 36 possible outputs); sweeping map size against test error found 13×13 as the actual sweet spot, past which training/test error diverge and overfitting sets in.
- **Semi-supervised** — the unsupervised phase uses all 546 unlabeled + labeled training points to learn the input topology, and only the supervised phase is restricted to the 39 labeled ones. This lets the map generalize to a much larger 20×20 grid without overfitting.

**Benchmarking.** The SOM wasn't just validated in isolation — it was benchmarked against four reference models on the same test set:

| Model | Test MAE | Test RMSE |
|---|---:|---:|
| Dummy Regressor | 0.311 | 0.378 |
| XGBoost | 0.175 | 0.212 |
| Random Forest | 0.167 | 0.187 |
| Bayesian Symbolic Regression | 0.127 | 0.142 |
| Fully-supervised SOM | 0.102 | 0.111 |
| **Semi-supervised SOM** | **0.061** | **0.084** |

The semi-supervised SOM cut test MAE by **~63% relative to Random Forest** — the strongest of the standard ML baselines — by exploiting the 91.5% of the dataset that had no label at all but was still informative about the shape of the input space.

**Beyond the metric: root-cause analysis.** Per-cluster permutation feature importance was run separately for each of the three high-degradation clusters the SOM surfaced. One cluster is dominated by cumulative flight exposure — the aircraft with the most flight cycles and flight hours in the fleet. The other two are both driven by a combination of manufacturing-stage features (independent of how the aircraft is flown — a possible quality-control signal) and environmental exposure, differing mainly in how much each factor contributes. This is the part of the study with the most direct engineering payoff — it's not just "here's the predicted degradation," it's "here's *why*," pointing Airbus's engineering teams toward specific, investigable causes rather than a black-box number.

**Impact.** The model reliably flags which aircraft will develop a degraded component, which is directly actionable for maintenance scheduling and parts provisioning. The feature-importance findings are now being investigated further with domain experts, with the ultimate goal of addressing the degradation at the design stage rather than just predicting it.

---

## What this project demonstrates

- Framing genuinely ambiguous, low-label, real-world problems as tractable ML tasks — both studies exist because the "obvious" supervised setup wasn't available (no ground-truth maturity labels; 91.5% of degradation labels missing)
- Extending a published academic method (SOM regression, Riese & Keller 2018) to a semi-supervised variant to survive a real data constraint, rather than just applying it off the shelf
- Rigorous baselining: every result is benchmarked against multiple alternatives (PCA vs. Kernel PCA; Dummy/XGBoost/Random Forest/Bayesian Symbolic Regression) and, for Study 1, against independent external industry reports — not just reported in isolation
- Explainability as a deliverable, not an afterthought: permutation feature importance was used to extract actionable, physically-grounded insight in both studies, not just to justify the model
- Working within real industrial constraints: confidential data handled correctly (anonymization), model outputs shipped as a monitored, non-technical-audience-facing dashboard, methodology co-designed with domain experts (marketing managers, maintenance engineers) rather than developed in isolation
- Tools: Python's ML/data-science stack (scikit-learn for baselines and dimensionality reduction, Optuna for hyperparameter optimization), Fuzzy C-Means, Kernel PCA, Self-Organizing Maps

## About this repository

This repository contains the full written thesis. The underlying code and datasets are proprietary to Airbus and are not publicly shareable; the report above documents the methodology, experiments, and results in full detail.
