# utils/helpers.py
import pandas as pd
import numpy as np

def validate_dataframe(df):
    errors = []
    if df.empty:          errors.append("DataFrame is empty")
    if df.shape[1] < 2:   errors.append("Need at least 2 columns")
    return errors

def get_df_summary(df):
    return {
        "shape":    df.shape,
        "missing":  int(df.isnull().sum().sum()),
        "dupes":    int(df.duplicated().sum()),
        "dtypes":   df.dtypes.value_counts().to_dict(),
    }