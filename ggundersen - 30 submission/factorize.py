"""Convert categorical variables to numbers.
"""

import pandas as pd


def factorize(df):
    """Convert features of type `object`, e.g. `string`, to categorical
    variables or factors.
    """
    for col in df.columns:
        if df[col].dtype == object:
            factors, values = pd.factorize(df[col])
            df[col] = factors
    return df
