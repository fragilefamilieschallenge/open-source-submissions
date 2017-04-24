"""Preprocessing script to perform principal component analysis on cleaned dataset.
"""

import pandas as pd
from sklearn.decomposition import PCA


def perform_pca(X, n_components, transform_and_save=False):
    """Perform principal component analysis on cleaned dataset_neg.
    """
    pca = PCA(n_components=n_components)
    pca.fit(X)
    if transform_and_save:
        data = pca.transform(X)
        # Cannot transfer columns since the number of features is reduced.
        X_pca = pd.DataFrame(data=data, index=X.index)
        print('Shape after PCA: (%s, %s)' % X_pca.shape)
        print('Saving file.')
        X_pca.to_pickle('data/private/X_pca_%s_components.p' % n_components)
    else:
        cumulative_variance = pca.explained_variance_ratio_.cumsum()[-1]
        print('Cumulative variance explained: %s' % cumulative_variance)
        return cumulative_variance


def find_ninety_nine_variance():
    """Perform principal component analysis with increasing number of
    components until 99% of variation is explained.
    """
    X = pd.read_pickle('data/private/X_cleaned.p')
    for n in range(1, X.shape[1]):
        print('PCA for %s components' % n)
        cumulative_variance = perform_pca(X, n)
        if cumulative_variance >= 0.99:
            perform_pca(X, n, transform_and_save=True)
            return


if __name__ == '__main__':
    find_ninety_nine_variance()
