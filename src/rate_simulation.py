
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
from config_loader import load_config

def main():
    """Main function to read PCA results and simulate yield curves."""
    config = load_config()
    processed_data = read_processed_data(config)
    pca_results = read_pca_results(config)
    simulated_curves = simulate_yield_curves(pca_results, processed_data)
    save_simulated_curves(config, simulated_curves)
    
def read_processed_data(config):
    """Reads the processed yield curve data from CSV file.
    Args:
        config: Configuration dictionary containing paths.
    Returns:
        pd.DataFrame: DataFrame containing the processed yield curve data.
    """
    processed_dir = config["data_directory"]["processed"]
    df = pd.read_csv(os.path.join(processed_dir, "cleaned_data.csv"), index_col=0, parse_dates=True)
    return df

def read_pca_results(config):
    """Reads PCA results from CSV files in the processed data directory.
    Args:
        config: Configuration dictionary containing paths.
    Returns:
        dict: A dictionary containing factors, loadings, and explained variance DataFrames.
    """
    processed_dir = config["data_directory"]["processed"]
    
    factors = pd.read_csv(os.path.join(processed_dir, "pca_factors.csv"), index_col=0)
    loadings = pd.read_csv(os.path.join(processed_dir, "pca_loadings.csv"), index_col=0)
    explained = pd.read_csv(os.path.join(processed_dir, "pca_explained_variance.csv"), index_col=0)
    
    return {
        "factors": factors,
        "loadings": loadings,
        "explained": explained
    }
    
def simulate_yield_curves(pca_results, processed_data, num_curves=10):
    """Simulates yield curves using PCA factors and loadings.
    Args:
        pca_results (dict): A dictionary containing factors, loadings, and explained variance DataFrames.
        processed_data (pd.DataFrame): DataFrame containing the processed yield curve data.
        num_curves (int): Number of yield curves to simulate.
    Returns:
        pd.DataFrame: A DataFrame containing the simulated yield curves.
    """
    factors = pca_results['factors']
    loadings = pca_results['loadings']
    mean_curve = processed_data.mean().values
    
    # Fit a VAR model to the PCA factors
    model = VAR(factors)
    results = model.fit(1)
    A = results.coefs[0]
    Sigma = results.sigma_u
    
    # Simulate new factor paths
    eps = np.random.multivariate_normal(np.zeros(Sigma.shape[0]), Sigma, size=num_curves)
    simulated_factors = (A @ factors.values[-1].reshape(-1, 1)).T + eps
    
    # Reconstruct yield curves from simulated factors
    simulated_curves_array = mean_curve + simulated_factors @ loadings.values.T
    simulated_curves = pd.DataFrame(simulated_curves_array, columns=processed_data.columns)
    
    return simulated_curves

def save_simulated_curves(config, simulated_curves):
    """Saves the simulated yield curves to a CSV file in the processed data directory.
    Args:
        config: Configuration dictionary containing paths.
        simulated_curves (pd.DataFrame): DataFrame containing the simulated yield curves.
    """
    processed_dir = config["data_directory"]["simulations"]
    os.makedirs(processed_dir, exist_ok=True)
    
    simulated_curves.to_csv(os.path.join(processed_dir, "simulated_yield_curves.csv"))
    
if __name__ == "__main__":
    main()