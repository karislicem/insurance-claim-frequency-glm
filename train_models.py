import numpy as np
import statsmodels.formula.api as smf

def train_poisson(df, formula):
    df = df.copy()
    df["log_exposure"] = np.log(df["Exposure"])

    model = smf.glm(
        formula=formula,
        data=df,
        family=smf.families.Poisson(),
        offset=df["log_exposure"]
    ).fit()

    return model


def train_two_stage(df, formula):

    df = df.copy()
    df["has_claim"] = (df["ClaimNb"] > 0).astype(int)

    # Stage 1
    logit_model = smf.logit(
        formula=formula.replace("ClaimNb", "has_claim"),
        data=df
    ).fit()

    # Stage 2
    df_pos = df[df["ClaimNb"] > 0].copy()
    df_pos["log_exposure"] = np.log(df_pos["Exposure"])

    poisson_model = smf.glm(
        formula=formula,
        data=df_pos,
        family=smf.families.Poisson(),
        offset=df_pos["log_exposure"]
    ).fit()

    return logit_model, poisson_model
