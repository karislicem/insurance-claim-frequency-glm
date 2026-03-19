import pandas as pd
import numpy as np

def create_features(df):
    df = df.copy()

    # Bands
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

    df["log_Density"] = np.log1p(df["Density"])

    return df
