import pandas as pd

def calibration_by_group(df, pred_col, group_col):

    result = (
        df.groupby(group_col)
        .agg({
            "ClaimNb": "sum",
            "Exposure": "sum",
            pred_col: "sum"
        })
    )

    result["actual_freq"] = result["ClaimNb"] / result["Exposure"]
    result["pred_freq"] = result[pred_col] / result["Exposure"]

    return result


def decile_analysis(df, pred_col):

    df = df.copy()
    df["decile"] = pd.qcut(df[pred_col], 10, labels=False)

    result = (
        df.groupby("decile")
        .agg({
            "ClaimNb": "sum",
            "Exposure": "sum",
            pred_col: "sum"
        })
    )

    result["actual_freq"] = result["ClaimNb"] / result["Exposure"]
    result["pred_freq"] = result[pred_col] / result["Exposure"]

    return result
