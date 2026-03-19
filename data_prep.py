import pandas as pd
import numpy as np

def load_data(path):
    return pd.read_csv(path)

def basic_clean(df):
    df = df.copy()
    df = df.drop(columns=["IDpol"], errors="ignore")
    return df
