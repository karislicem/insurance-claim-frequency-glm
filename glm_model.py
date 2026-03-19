import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ================================
# LOAD DATA
# ================================
df = pd.read_csv("freMTPL2freq.csv")  # Change this if needed

df = df.drop(columns=["IDpol"], errors="ignore")

# ================================
# BASIC FEATURES
# ================================
df = df[df["Exposure"] > 0]

df["claim_freq"] = df["ClaimNb"] / df["Exposure"]
df["log_exposure"] = np.log(df["Exposure"])
df["has_claim"] = (df["ClaimNb"] > 0).astype(int)
df["log_Density"] = np.log1p(df["Density"])

# ================================
# BANDING
# ================================
df["DrivAge_band"] = pd.cut(
    df["DrivAge"],
    bins=[18, 25, 35, 50, 65, 100],
    labels=["18-25", "25-35", "35-50", "50-65", "65+"]
)

df["VehAge_band"] = pd.cut(
    df["VehAge"],
    bins=[0, 2, 5, 10, 20, 100],
    labels=["0-2", "2-5", "5-10", "10-20", "20+"]
)

df["BonusMalus_band"] = pd.cut(
    df["BonusMalus"],
    bins=[0, 50, 75, 100, 150, 300],
    labels=["<=50", "50-75", "75-100", "100-150", "150+"]
)

# ================================
# BASELINE
# ================================
print("\nBASELINE FREQUENCY")
baseline = df["ClaimNb"].sum() / df["Exposure"].sum()
print(baseline)

# ================================
# POISSON MODEL
# ================================
formula = """
ClaimNb ~
C(Region) +
C(VehGas) +
C(VehBrand) +
C(BonusMalus_band) +
C(DrivAge_band) +
C(VehAge_band) +
VehPower +
log_Density
"""

poisson_model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Poisson(),
    offset=df["log_exposure"]
).fit()

df["pred_poisson"] = poisson_model.predict(df)

print("\nPOISSON SUMMARY")
print(poisson_model.summary())

# ================================
# POISSON CALIBRATION (AREA)
# ================================
print("\nPOISSON CALIBRATION BY AREA")

poisson_cal = df.groupby("Area").agg({
    "ClaimNb": "sum",
    "Exposure": "sum",
    "pred_poisson": "sum"
})

poisson_cal["actual_freq"] = poisson_cal["ClaimNb"] / poisson_cal["Exposure"]
poisson_cal["pred_freq"] = poisson_cal["pred_poisson"] / poisson_cal["Exposure"]

print(poisson_cal)

# ================================
# NEGATIVE BINOMIAL
# ================================
nb_model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.NegativeBinomial(),
    offset=df["log_exposure"]
).fit()

df["pred_nb"] = nb_model.predict(df)

print("\nNEGATIVE BINOMIAL SUMMARY")
print(nb_model.summary())

# ================================
# TWO-STAGE MODEL
# ================================

# Step 1: Logistic
logit_model = smf.logit(
    formula=formula.replace("ClaimNb", "has_claim"),
    data=df
).fit()

df["p_claim"] = logit_model.predict(df)

# Step 2: Frequency (only positives)
df_pos = df[df["ClaimNb"] > 0].copy()

freq_model = smf.glm(
    formula=formula,
    data=df_pos,
    family=sm.families.Poisson(),
    offset=df_pos["log_exposure"]
).fit()

df["lambda_given_claim"] = freq_model.predict(df)

# Final prediction
df["final_pred"] = df["p_claim"] * df["lambda_given_claim"]

# ================================
# FINAL CALIBRATION
# ================================
print("\nFINAL MODEL CALIBRATION (AREA)")

final_cal = df.groupby("Area").agg({
    "ClaimNb": "sum",
    "Exposure": "sum",
    "final_pred": "sum"
})

final_cal["actual_freq"] = final_cal["ClaimNb"] / final_cal["Exposure"]
final_cal["pred_freq"] = final_cal["final_pred"] / final_cal["Exposure"]

print(final_cal)

# ================================
# DECILE ANALYSIS
# ================================
print("\nDECILE ANALYSIS")

df["decile"] = pd.qcut(df["final_pred"], 10, labels=False)

decile = df.groupby("decile").agg({
    "ClaimNb": "sum",
    "Exposure": "sum",
    "final_pred": "sum"
})

decile["actual_freq"] = decile["ClaimNb"] / decile["Exposure"]
decile["pred_freq"] = decile["final_pred"] / decile["Exposure"]

print(decile)
