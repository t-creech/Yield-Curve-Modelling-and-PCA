
import pandas as pd
import numpy as np
import os
from statsmodels.tsa.api import VAR
from config_loader import load_config

# Set random seed for reproducibility
np.random.seed(42)

def main():
    """Main function to read PCA results and simulate yield curves."""
    config = load_config()
    processed_data = read_processed_data(config)
    pca_results = read_pca_results(config)
    simulated_curves = simulate_yield_curves(pca_results, processed_data, config)
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
    
    return {
        "factors": factors,
        "loadings": loadings,
    }
    
def simulate_yield_curves(pca_results, processed_data, config):
    """Simulates yield curves using PCA factors and loadings based on a VAR(1) model.
    Args:
        pca_results (dict): A dictionary containing factors, loadings, and explained variance DataFrames.
        processed_data (pd.DataFrame): DataFrame containing the processed yield curve data.
        config: Configuration dictionary containing simulation parameters.
    Returns:
        pd.DataFrame: A DataFrame containing the simulated yield curves.
    """
    factors = pca_results['factors']
    loadings = pca_results['loadings']
    cleaned_yield_curves = processed_data
    
    # Get simulation parameters
    n_simulations = config["num_simulations"]
    n_steps = config["simulation_horizon_days"]
    VAR_order = config["VAR_order"]

    # Fit a VAR model to the PCA factors
    model = VAR(factors.values)
    results = model.fit(VAR_order, trend="n")
    A = results.coefs
    Sigma = results.sigma_u
    initial_state = factors.values[-1, :]

    simulated_factors_changes = np.zeros((n_simulations, n_steps + 1, factors.shape[1]))
    
    # Set initial state for all simulations
    simulated_factors_changes[:, 0, :] = initial_state
    
    # Run simulations
    for sim in range(n_simulations):
        for step in range(1, n_steps + 1):
            # Sample epsilon from multivariate normal
            epsilon = np.random.multivariate_normal(mean=np.zeros(factors.shape[1]), cov=Sigma)
            # Update state
            new_state = np.zeros(factors.shape[1])
            for lag in range(VAR_order):
                new_state += A[lag] @ simulated_factors_changes[sim, step - lag - 1, :]
            new_state += epsilon
            simulated_factors_changes[sim, step, :] = new_state

    # Remove the initial state from the simulations
    simulated_factors_changes = simulated_factors_changes[:, 1:, :]
    
    simulated_diff_yields = simulated_factors_changes @ loadings.values.T
    simulated_cumulative_yields = np.cumsum(simulated_diff_yields, axis=1)
    base_curve = cleaned_yield_curves.values[-1]
    simulated_yields = simulated_cumulative_yields + base_curve
    
    last_date = cleaned_yield_curves.index[-1]
    simulated_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_steps, freq='B')
    
    dfs = []
    for sim in range(n_simulations):
        df = pd.DataFrame(simulated_yields[sim], columns=loadings.index)
        df.index = simulated_dates
        df["sim_id"] = sim
        dfs.append(df)

    simulated_curves = pd.concat(dfs).set_index("sim_id", append=True)

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