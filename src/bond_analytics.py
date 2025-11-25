
import numpy as np
import pandas as pd
from datetime import timedelta
import calendar

def is_business_day(date):
    """Checks if a given date is a business day (Monday to Friday).
    Args:
        date (pd.Timestamp): The date to check.
    Returns:
        bool: True if the date is a business day, False otherwise.
    """
    return date.weekday() < 5

def adjust_to_business_day(date, convention="following"):
    """Adjusts a date to the nearest business day based on the specified convention.
    Args:
        date (pd.Timestamp): The date to adjust.
        convention (str): The business day convention ("following" or "preceding").
    Returns:
        pd.Timestamp: The adjusted business day.
    """
    if is_business_day(date):
        return date

    if convention == "following":
        while not is_business_day(date):
            date += timedelta(days=1)
    elif convention == "modified_following":
        original_month = date.month
        while not is_business_day(date):
            date += timedelta(days=1)
        if date.month != original_month:
            date -= timedelta(days=1)
            while not is_business_day(date):
                date -= timedelta(days=1)
    elif convention == "preceding":
        while not is_business_day(date):
            date -= timedelta(days=1)
    else:
        raise ValueError("Unsupported business day convention.")
    
    return date

def year_fraction(start_date, end_date, day_count_convention="30/360"):
    """Calculates the year fraction between two dates based on the specified day count convention.
    Args:
        start_date (pd.Timestamp): The start date.
        end_date (pd.Timestamp): The end date.
        day_count_convention (str): The day count convention to use ("30/360", "ACT/360", "ACT/365", "ACT/ACT").
    Returns:
        float: The year fraction between the two dates.
    """
    if day_count_convention.upper() == "30/360":
        d1 = start_date.day
        d2 = end_date.day
        m1 = start_date.month
        m2 = end_date.month
        y1 = start_date.year
        y2 = end_date.year

        if d1 == 31:
            d1 = 30
        if d2 == 31:
            d2 = 30

        return ((360 * (y2 - y1)) + (30 * (m2 - m1)) + (d2 - d1)) / 360.0

    elif day_count_convention.upper() == "ACT/360":
        return (end_date - start_date).days / 360.0

    elif day_count_convention.upper() == "ACT/365":
        return (end_date - start_date).days / 365.0

    elif day_count_convention.upper() == "ACT/ACT":
        yf = 0
        current = start_date
        while current < end_date:
            year_end = pd.Timestamp(year=current.year, month=12, day=31)
            next_period_end = min(year_end, end_date)
            days_in_period = (next_period_end - current).days + 1
            yf += days_in_period / (366 if calendar.isleap(current.year) else 365)
            current = next_period_end + timedelta(days=1)
        return yf

    else:
        raise ValueError("Unsupported day count convention.")

def tenor_to_years(tenor):
    """Converts a tenor string to its equivalent in years.
    Args:
        tenor (str): The tenor string (e.g., "6MO", "1YR", "2Y").
    Returns:
        float: The equivalent tenor in years.
    """
    t = tenor.upper()
    if t.endswith("MO"):
        months = int(t[:-2])
        return months / 12.0
    elif t.endswith("YR"):
        years = int(t[:-2])
        return years
    elif t.endswith("Y"):
        years = int(t[:-1])
        return years
    else:
        raise ValueError(f"Unsupported tenor format: {tenor}")

def generate_cashflows(settlement_date, maturity_date, coupon_rate, frequency = 2, face_value=100, business_day_convention="following"):
    """Generates cashflows for a bond.
    Args:
        settlement_date (pd.Timestamp): The settlement date of the bond.
        maturity_date (pd.Timestamp): The maturity date of the bond.
        coupon_rate (float): The annual coupon rate as a decimal.
        frequency (int): Number of coupon payments per year.
        face_value (float): The face value of the bond. Default is 100.
        business_day_convention (str): The business day convention to use. Default is "following".
    Returns:
        pd.DataFrame: DataFrame containing the cashflow schedule.
    """
    settlement_date = pd.Timestamp(settlement_date)
    maturity_date = pd.Timestamp(maturity_date)
    
    months_between_coupons = 12 // frequency
    cashflow_dates = []
    current_date = maturity_date
    
    while current_date > settlement_date:
        cashflow_dates.append(current_date)
        year = current_date.year
        month = current_date.month - months_between_coupons
        if month <= 0:
            month += 12
            year -= 1
        day = min(current_date.day, calendar.monthrange(year, month)[1])
        current_date = pd.Timestamp(year=year, month=month, day=day)

    cashflow_dates = sorted(cashflow_dates)
    cashflow_dates = [adjust_to_business_day(date, business_day_convention) for date in cashflow_dates]
    
    coupon = (coupon_rate / frequency) * face_value
    cashflows = np.full(len(cashflow_dates), coupon)
    cashflows[-1] += face_value  # Add face value to the last cashflow

    return pd.DataFrame({
        "date": cashflow_dates,
        "cashflow_amount": cashflows,
    }, index=cashflow_dates)

def discount_factors(cashflow_dates, curve, day_count_convention="ACT/365"):
    """Builds discount factors from a yield curve by interpolating yields to cashflow dates.
    Args:
        cashflow_dates (list of pd.Timestamp): List of cashflow dates.
        curve (pd.DataFrame): DataFrame containing the yield curve with tenors and yields for a specific date.
        day_count_convention (str): The day count convention to use for discounting. Default is "ACT/365".
    Returns:
        pd.DataFrame: DataFrame containing discount factors for the cashflow dates.
    """
    
    if "date" not in curve.columns:
        # if the date is the index, promote it to a column
        if curve.index.name == "date":
            curve = curve.reset_index()
        else:
            raise ValueError("curve must have a 'date' column or a named date index")
    
    curve_long = curve.melt(id_vars="date", var_name="tenor", value_name="yield")
    curve_date = pd.to_datetime(curve_long["date"].iloc[0])    
    curve_long["tenor_length"] = curve_long["tenor"].apply(tenor_to_years)
    curve_long = curve_long.sort_values(by="tenor_length")
    tenors = curve_long["tenor_length"].to_numpy(float)
    curve_yields = curve_long["yield"].to_numpy(float)
    
    cashflow_dates = sorted(cashflow_dates)
    
    t = np.array([year_fraction(curve_date, date, day_count_convention) for date in cashflow_dates])

    tenors = np.asarray(tenors)
    curve_yields = np.asarray(curve_yields)

    y_interp = np.interp(t, tenors, curve_yields)

    discount_factors = pd.DataFrame(index=cashflow_dates)
    discount_factors["discount_factor"] = np.exp(-y_interp * t)
    discount_factors["year_fraction"] = t

    return discount_factors

def price_duration_convexity(curve, settlement_date, maturity_date, coupon_rate, frequency=2, face_value=100, business_day_convention="following", day_count_convention="ACT/365"):
    """Calculates the price, duration, and convexity of a bond given a yield curve and bond parameters.
    Args:
        curve (pd.DataFrame): DataFrame containing the yield curve with tenors and yields for a specific date.
        settlement_date (pd.Timestamp): The settlement date of the bond.
        maturity_date (pd.Timestamp): The maturity date of the bond.
        coupon_rate (float): The annual coupon rate as a decimal.
        frequency (int): Number of coupon payments per year.
        face_value (float): The face value of the bond. Default is 100.
        business_day_convention (str): The business day convention to use. Default is "following".
        day_count_convention (str): The day count convention to use for discounting. Default is "ACT/365".
    Returns:
        dict: A dictionary containing the bond price, modified duration, and convexity.
    """
    
    cash_flows_df = generate_cashflows(settlement_date, maturity_date, coupon_rate, frequency, face_value, business_day_convention)
    
    discount_factors_df = discount_factors(cash_flows_df["date"].tolist(), curve, day_count_convention)
    t = discount_factors_df["year_fraction"].values
    df = discount_factors_df["discount_factor"].values
    cf = cash_flows_df["cashflow_amount"].values
    
    present_values = cf * df
    price = present_values.sum()
    
    macaulay_duration = np.sum(present_values * t) / price
    y_eff = np.average(-np.log(df) / t, weights=df)
    modified_duration = macaulay_duration / (1 + y_eff / frequency)
    
    convexity = np.sum(present_values * t**2) / price
    
    return {
        "price": price,
        "macaulay_duration": macaulay_duration,
        "modified_duration": modified_duration,
        "convexity": convexity,
        "cash_flows": cash_flows_df
    }
