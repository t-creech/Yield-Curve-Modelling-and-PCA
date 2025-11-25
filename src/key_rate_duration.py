import numpy as np
import pandas as pd
from copy import deepcopy
from bond_analytics import price_duration_convexity

def shock_single_tenor(curve, tenor, shock_size_bp):
    """Applies a shock to a single tenor in the yield curve.
    Args:
        curve (pd.DataFrame): DataFrame containing the yield curve with tenors and yields for a specific date.
        tenor (str): The tenor to which the shock will be applied (e.g., "5Y").
        shock_size_bp (float): The size of the shock in basis points.
    Returns:
        pd.DataFrame: A new DataFrame with the shocked yield curve.
    """
    shocked_curve = deepcopy(curve)
    if tenor in shocked_curve.columns:
        shocked_curve[tenor] += shock_size_bp / 10000.0
    else:
        raise ValueError(f"Tenor {tenor} not found in the yield curve.")
    
    return shocked_curve

def compute_key_rate_duration(curve, tenor, settlement_date, maturity_date, coupon_rate, frequency=2, face_value=100, business_day_convention="following", day_count_convention="ACT/365", shock_size_bp=1.0):
    """Computes the key rate duration of a bond given a yield curve and bond parameters.
    Args:
        curve (pd.DataFrame): DataFrame containing the yield curve with tenors and yields for a specific date.
        tenor (str): The tenor to which the shock will be applied (e.g., "5Y").
        settlement_date (pd.Timestamp): The settlement date of the bond.
        maturity_date (pd.Timestamp): The maturity date of the bond.
        coupon_rate (float): The annual coupon rate as a decimal.
        frequency (int): Number of coupon payments per year.
        face_value (float): The face value of the bond. Default is 100.
        business_day_convention (str): The business day convention to use. Default is "following".
        day_count_convention (str): The day count convention to use. Default is "ACT/365".
        shock_size_bp (float): The size of the shock in basis points. Default is 1.0.
    Returns:
        float: The key rate duration of the bond.
    """
    if shock_size_bp == 0:
        raise ValueError("shock_size_bp must be non-zero to compute key rate duration.")
    P0 = price_duration_convexity(curve, settlement_date, maturity_date, coupon_rate, frequency, face_value, business_day_convention, day_count_convention)["price"]
    shocked_curve = shock_single_tenor(curve, tenor, shock_size_bp)
    P_plus = price_duration_convexity(shocked_curve, settlement_date, maturity_date, coupon_rate, frequency, face_value, business_day_convention, day_count_convention)["price"]
    shocked_curve = shock_single_tenor(curve, tenor, -shock_size_bp)
    P_minus = price_duration_convexity(shocked_curve, settlement_date, maturity_date, coupon_rate, frequency, face_value, business_day_convention, day_count_convention)["price"]
    key_rate_duration = (P_minus - P_plus) / (2 * (shock_size_bp / 10000.0) * P0)
    return key_rate_duration

def compute_krd_vector(curve, tenors, settlement_date, maturity_date, coupon_rate, frequency=2, face_value=100, business_day_convention="following", day_count_convention="ACT/365", shock_size_bp=1.0):
    """Computes the key rate duration vector for multiple tenors.
    Args:
        curve (pd.DataFrame): DataFrame containing the yield curve with tenors and yields for a specific date.
        tenors (list of str): List of tenors to compute key rate durations for (e.g., ["1Y", "2Y", "5Y"]).
        settlement_date (pd.Timestamp): The settlement date of the bond.
        maturity_date (pd.Timestamp): The maturity date of the bond.
        coupon_rate (float): The annual coupon rate as a decimal.
        frequency (int): Number of coupon payments per year.
        face_value (float): The face value of the bond. Default is 100.
        business_day_convention (str): The business day convention to use. Default is "following".
        day_count_convention (str): The day count convention to use. Default is "ACT/365".
        shock_size_bp (float): The size of the shock in basis points. Default is 1.0.
    Returns:
        dict: A dictionary where keys are tenors and values are the corresponding key rate durations.
    """
    krd_vector = {}
    for tenor in tenors:
        krd = compute_key_rate_duration(curve, tenor, settlement_date, maturity_date, coupon_rate, frequency, face_value, business_day_convention, day_count_convention, shock_size_bp)
        krd_vector[tenor] = krd
    return krd_vector

DEFAULT_KRD_TENORS = ["1Y", "2Y", "5Y", "10Y", "30Y"]

def prepare_krd_for_plot(krd_vector, tenors=DEFAULT_KRD_TENORS):
    """Prepares key rate duration vectors for plotting.
    Args:
        krd_vector (dict): A dictionary where keys are tenors and values are the corresponding key rate durations.
        tenors (list of str): List of tenors to include in the plot. Default is DEFAULT_KRD_TENORS.
    Returns:
        pd.DataFrame: A DataFrame suitable for plotting key rate durations.
    """
    return pd.DataFrame.from_dict(krd_vector, orient="index").loc[tenors]