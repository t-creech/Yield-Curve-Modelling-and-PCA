import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def run_pca(df: pd.DataFrame, n_components: int = 3):
    """Applies PCA to the given DataFrame and returns the principal components, loadings, and explained variance.
    Args:
        df (pd.DataFrame): The DataFrame containing the data to apply PCA on.
        n_components (int): The number of principal components to compute.
    Returns:
        factors_df (pd.DataFrame): DataFrame containing the principal components.
        loadings (pd.DataFrame): DataFrame containing the loadings for each original variable.
        explained (np.ndarray): Array containing the explained variance ratio for each principal component.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    pca = PCA(n_components=n_components)
    factors = pca.fit_transform(X)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=df.columns,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    factors_df = pd.DataFrame(
        factors,
        index=df.index,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    explained = pca.explained_variance_ratio_

    return factors_df, loadings, explained