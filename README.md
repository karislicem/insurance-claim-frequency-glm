# Motor Insurance Claim Frequency Modeling with GLMs

## Project Overview

This project focuses on modeling motor insurance claim frequency using Generalized Linear Models (GLMs). The goal is to estimate expected claim frequency per policy while accounting for exposure and risk characteristics.

The project emphasizes interpretability and actuarial relevance rather than purely maximizing predictive performance.

---

## Dataset

The dataset contains motor insurance policy-level data, including:

* **ClaimNb**: Number of claims (target)
* **Exposure**: Policy exposure duration
* **Risk factors**:

  * Driver age, vehicle age, vehicle power
  * Bonus-malus score
  * Vehicle brand and fuel type
  * Geographic indicators (Area, Region)
  * Population density

---

## Problem Definition

We aim to model expected claim frequency:

E[ClaimNb] = Exposure × λ

Using a log-link GLM:

log(E[ClaimNb]) = log(Exposure) + Xβ

Where:

* `log(Exposure)` is used as an offset
* `λ` represents claim frequency per unit exposure

---

## Methodology

The modeling process follows these steps:

1. Exploratory data analysis (EDA)
2. Feature engineering:

   * Banding of key variables (Driver Age, Vehicle Age, Bonus-Malus)
   * Log transformation of density
3. Baseline Poisson GLM with exposure offset
4. Negative Binomial GLM to address overdispersion
5. Two-stage modeling approach:

   * Logistic regression for claim occurrence
   * Poisson GLM for claim count given a claim
6. Model evaluation:

   * Calibration by groups
   * Decile-based analysis

---

## Model Development

### 1. Poisson GLM

A standard Poisson model was first used with exposure as an offset. This model provided a baseline but showed systematic overestimation.

### 2. Negative Binomial GLM

To address overdispersion in the data, a Negative Binomial model was implemented. While this improved fit, calibration issues remained.

### 3. Two-Stage Model (Final Model)

Due to a high proportion of zero-claim observations, a two-stage approach was used:

* **Stage 1**: Logistic regression (probability of having at least one claim)
* **Stage 2**: Poisson GLM (expected number of claims given a claim)

Final prediction:

Expected Claims = P(Claim > 0) × E[ClaimNb | Claim > 0]

This approach improved calibration and better reflects the structure of insurance data.

---

## Results

* The model shows strong risk segmentation across deciles
* Claim frequency increases consistently from lowest to highest risk groups
* Some overestimation remains, but ranking performance is strong
* Bonus-malus is the most influential predictor
* Driver age and regional factors also contribute significantly

---

## Key Takeaways

* Exposure must be included as an offset in claim modeling
* Claim data often exhibits overdispersion and zero inflation
* Two-stage models can improve calibration in insurance datasets
* GLMs provide interpretable insights for pricing decisions

---

## Repository Structure

* `notebooks/`: end-to-end workflow
* `src/`: modular code for preprocessing, modeling, and evaluation
* `outputs/`: generated tables and figures

---

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the notebook:

```bash
notebooks/01_glm_model.ipynb
```

---

## Notes

This project is intended as a learning-oriented exploration of GLMs in an insurance context. The focus is on understanding modeling choices and their impact on results.
