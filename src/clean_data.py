import pandas as pd
import os
from config_loader import load_config

def main():
    """Main function to clean the combined data CSV file and save the cleaned data."""
    
    config = load_config()
    cleaned_dir = config["data_directory"]["processed"]
    os.makedirs(cleaned_dir, exist_ok=True)
    
    # Clean the data
    cleaned_data = clean_data(config)
    
    # Save the cleaned data to a new CSV file
    save_cleaned_data(cleaned_data, cleaned_dir)
    
    # Compute yield changes
    cleaned_data_diffs = compute_yield_changes(cleaned_data)
    
    # Save the cleaned diffs to a new CSV file
    save_cleaned_diffs(cleaned_data_diffs, cleaned_dir)

def clean_data(config):
    """Reads data from the combined CSV file, ensure the columns being read are expected, and cleans the data.
    Args:
        config: Configuration dictionary containing paths.
    Returns:
        pd.DataFrame: A cleaned DataFrame with the expected columns.
    """
    path = os.path.join(config["data_directory"]["raw"], "combined_data.csv")
    data = pd.read_csv(path, parse_dates=['date'])
    expected_columns = ['date'] + list(config['tenors'].values())
    if not all(col in data.columns for col in expected_columns):
        raise ValueError(f"Data does not contain all expected columns: {expected_columns}")
    # Ensure the date column is in datetime format
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    # Drop rows with NaN values in the data columns
    data.dropna(subset=expected_columns[1:], inplace=True)
    # Reset index for the cleaned DataFrame
    data = data.set_index('date')
    return data

def compute_yield_changes(df):
    """Computes daily changes in yields for a specified tenor.
    Args:
        df (pd.DataFrame): The DataFrame containing yield data.
    Returns:
        pd.Series: A Series containing the daily changes in yields.
    """
    diffs = df.diff().dropna()
    return diffs

def save_cleaned_data(df, path):
    """Saves the cleaned DataFrame to a CSV file.
    Args:
        df (pd.DataFrame): The cleaned DataFrame to save.
        path (str): The file path where the CSV should be saved.
    """
    csv_path = os.path.join(path, 'cleaned_data.csv')
    df.to_csv(csv_path, index=True)
    print(f"Cleaned data saved to {csv_path}")

def save_cleaned_diffs(df, path):
    """Saves the cleaned diffs DataFrame to a CSV file.
    Args:
        df (pd.DataFrame): The cleaned diffs DataFrame to save.
        path (str): The file path where the CSV should be saved.
    """
    csv_path = os.path.join(path, 'cleaned_data_diffs.csv')
    df.to_csv(csv_path, index=True)
    print(f"Cleaned data diffs saved to {csv_path}")

if __name__ == "__main__":
    main()
