"""Impute missing values:

* Columns of dtype object are imputed with the most frequent value in column.
* Columns of other types are imputed with mean of column.

Credit: http://stackoverflow.com/a/25562948/1830334
"""

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin


class DataFrameImputer(TransformerMixin):

    def fit(self, X):
        data = []
        for col in X.columns:
            is_categorical = X[col].dtype == np.dtype('O')
            if is_categorical:
                data.append(X[col].value_counts().index[0])
            else:
                data.append(X[col].mean())
        self.fill = pd.Series(data, index=X.columns)
        return self

    def transform(self, X):
        X = X.fillna(self.fill)
        return X
