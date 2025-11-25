from config_loader import load_config
import numpy as np
import pandas as pd
import os
from key_rate_duration import compute_krd_vector, DEFAULT_KRD_TENORS
from bond_analytics import price_duration_convexity


def extract_sim_curve_on_date(sim_data, path_id, date):
    """Extracts the yield curve for a specific date.
    Args:
        sim_data (pd.DataFrame): DataFrame containing the simulated yield curves.
        path_id: The path ID for the simulation.
        date (pd.Timestamp): The date for which to extract the yield curve.
    Returns:
        pd.DataFrame: A DataFrame containing the simulated yield curve for the specified date.
    """

    if date not in sim_data.index.get_level_values(0):
        raise ValueError(f"Date {date} not found in both cleaned and simulated yield curves.")
    if path_id not in sim_data.index.get_level_values(1):
        raise ValueError(f"Path ID {path_id} not found in simulated yield curves.")
    
    curve = sim_data.loc[[(date, path_id)]].reset_index(level=1, drop=True)
    curve.index.name = "date"
    
    return curve

def compute_mc_analytics(config, date, settlement_date, maturity_date, coupon_rate, frequency=2, face_value=100, business_day_convention="following", day_count_convention="ACT/365", shock_size_bp=1.0):
    """Computes bond analytics across all simulated yield curves for specified date.
    Args:
        config: Configuration dictionary containing paths.
        date (pd.Timestamp): The date for which to compute bond analytics.
        settlement_date (pd.Timestamp): The settlement date of the bond.
        maturity_date (pd.Timestamp): The maturity date of the bond.
        coupon_rate (float): The annual coupon rate as a decimal.
        frequency (int): Number of coupon payments per year.
        face_value (float): The face value of the bond. Default is 100.
        business_day_convention (str): The business day convention to use. Default is "following".
        day_count_convention (str): The day count convention to use. Default is "ACT/365".
        shock_size_bp (float): The size of the shock in basis points. Default is 1.0.
    Returns:
        pd.DataFrame: A DataFrame containing bond analytics for each simulation path.
    """
    sim_curves_path = os.path.join(config["data_directory"]["simulations"], "simulated_yield_curves.csv")
    curve_df = pd.read_csv(sim_curves_path, index_col=[0,1], parse_dates=[0])
    sim_ids = curve_df.index.get_level_values(1).unique()
    
    results = []
    for sim_id in sim_ids:
        curve_df_single = extract_sim_curve_on_date(curve_df, sim_id, date)
        price_duration_convexity_res = price_duration_convexity(curve_df_single, settlement_date, maturity_date, coupon_rate, frequency, face_value, business_day_convention, day_count_convention)
        krd_vector = compute_krd_vector(curve_df_single, DEFAULT_KRD_TENORS, settlement_date, maturity_date, coupon_rate, frequency, face_value, business_day_convention, day_count_convention, shock_size_bp)
        
        result = {
            "date": date,
            "sim_id": sim_id,
            "price": price_duration_convexity_res["price"],
            "modified_duration": price_duration_convexity_res["modified_duration"],
            "convexity": price_duration_convexity_res["convexity"],
        }
        for tenor, krd in krd_vector.items():
            result[f"krd_{tenor}"] = krd
        results.append(result)
        
    return pd.DataFrame(results)

