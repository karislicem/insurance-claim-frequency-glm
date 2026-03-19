# Motor Insurance Claim Frequency Modeling with GLMs

## Overview

This project explores how Generalized Linear Models (GLMs) can be used to model insurance claim frequency.

The main goal is to estimate expected claim frequency per policy while taking into account exposure and key risk factors. The focus is on understanding the modeling process and interpreting results rather than building a highly optimized model.

---

## Dataset

The dataset contains motor insurance policy-level data with the following key variables:

* **ClaimNb**: Number of claims (target)
* **Exposure**: Duration of exposure for each policy
* **Driver and vehicle characteristics**:

  * Driver age (`DrivAge`)
  * Vehicle age (`VehAge`)
  * Vehicle power (`VehPower`)
  * Bonus-malus score (`BonusMalus`)
* **Categorical features**:

  * Area, Region
  * Vehicle brand (`VehBrand`)
  * Fuel type (`VehGas`)
* **Density**: Population density proxy

---

## Exploratory Data Analysis

Below are the distributions of the main numerical features:

![Numerical Features Distribution](claim_hist.png)

Key observations:

* Claim counts are highly skewed with a large number of zero-claim policies
* Exposure is concentrated around lower values but varies across policies
* Density is heavily right-skewed
* Bonus-malus shows strong clustering around lower values

---

## Problem Definition

We model claim frequency using a standard insurance formulation:

E[ClaimNb] = Exposure × λ

Using a log-link GLM:

log(E[ClaimNb]) = log(Exposure) + Xβ

Here, `log(Exposure)` is used as an offset.

---

## Modeling Approach

### 1. Poisson GLM

A Poisson model was first fitted using exposure as an offset.

* Simple and interpretable
* However, it showed systematic overestimation

---

### 2. Negative Binomial GLM

To address overdispersion, a Negative Binomial model was tested.

* Slight improvement in fit
* Calibration issues still remained

---

### 3. Two-Stage Model (Final Model)

Due to the high number of zero-claim observations, a two-stage approach was used:

* **Stage 1**: Logistic regression to model the probability of having at least one claim
* **Stage 2**: Poisson GLM to model the number of claims given a claim occurred

Final prediction:

Expected Claims = P(Claim > 0) × E[ClaimNb | Claim > 0]

This approach produced better calibration and more realistic predictions.

---

## Results

### Calibration by Area (Final Model)

The two-stage model reduces the overestimation observed in single-stage models and produces more stable results across different regions.

---

### Decile Analysis

The model shows clear separation between low-risk and high-risk groups:

* Claim frequency increases consistently across deciles
* Higher predicted risk corresponds to higher observed claim frequency

This indicates good ranking performance.

---

## Key Findings

* Bonus-malus is the strongest predictor of claim frequency
* Younger and older drivers tend to have higher risk
* Density has a positive relationship with claim frequency
* Significant variation exists across regions
* Two-stage modeling improves calibration compared to standard GLMs

---

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Place the dataset in the project folder (Don't forget to extract from zip)

3. Run the script:

```bash
python insurance_claim_frequency_glm.py
```

---

## Notes

This project is intended as a learning-oriented implementation of GLMs in an insurance context. The emphasis is on understanding modeling choices and interpreting results rather than building a production-ready model.
