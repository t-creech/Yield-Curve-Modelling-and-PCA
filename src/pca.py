import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from config_loader import load_config

def main():
    """Main function to run PCA on the cleaned data and save the results."""
    config = load_config()
    
    df = read_data(config)
    factors, loadings, explained = run_pca(df, n_components=3)
    save_pca_results(config, factors, loadings, explained)

    print("PCA complete. Variance explained:", explained)

def save_pca_results(config, factors: pd.DataFrame, loadings: pd.DataFrame, explained: pd.DataFrame):
    """Saves PCA results to CSV files in the processed data directory.
    Args:
        config: Configuration dictionary containing paths.
        factors (pd.DataFrame): DataFrame containing the principal components.
        loadings (pd.DataFrame): DataFrame containing the loadings for each original variable.
        explained (pd.DataFrame): DataFrame containing the explained variance ratio for each principal component.
    """
    processed_dir = config["data_directory"]["processed"]
    os.makedirs(processed_dir, exist_ok=True)

    factors.to_csv(os.path.join(processed_dir, "pca_factors.csv"))
    
    loadings.index.name = "tenor"
    loadings.to_csv(os.path.join(processed_dir, "pca_loadings.csv"))

    explained.index.name = "PC"
    explained.to_csv(os.path.join(processed_dir, "pca_explained_variance.csv"))

def run_pca(df: pd.DataFrame, n_components: int = 3):
    """Applies PCA to the given DataFrame and returns the principal components, loadings, and explained variance.
    Args:
        df (pd.DataFrame): The DataFrame containing the data to apply PCA on.
        n_components (int): The number of principal components to compute.
    Returns:
        factors_df (pd.DataFrame): DataFrame containing the principal components.
        loadings (pd.DataFrame): DataFrame containing the loadings for each original variable.
        explained (pd.DataFrame): DataFrame containing the explained variance ratio for each principal component.
    """
    X = df.values

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
    explained_df = pd.DataFrame(
        explained,
        index=[f"PC{i+1}" for i in range(len(explained))],
        columns=["Explained_Variance_Ratio"]
    )
    
    # Ensure consistent sign for loadings
    for col in loadings.columns:
        if loadings[col].sum() < 0:
            loadings[col] = -loadings[col]
            factors_df[col] = -factors_df[col]


    return factors_df, loadings, explained_df

def read_data(config):
    """Reads the cleaned data from the processed data directory.
    Args:
        config: Configuration dictionary containing paths.
    Returns:
        pd.DataFrame: DataFrame containing the cleaned data.
    """
    processed_dir = config["data_directory"]["processed"]
    path = os.path.join(processed_dir, 'cleaned_data_diffs.csv')
    df = pd.read_csv(path, parse_dates=['date']).set_index('date')
    return df

if __name__ == "__main__":
    main()
