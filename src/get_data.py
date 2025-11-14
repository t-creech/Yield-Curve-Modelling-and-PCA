## Script to pull data from FRED and save it to a CSV file in the data directory.
import os
import pandas as pd
import requests
import logging
from datetime import datetime
from config_loader import load_config

def main():
    """Main function to fetch data from FRED and save to CSV files."""
    # Load configuration
    config = load_config()

    # Fetch all data
    combined_df = fetch_all_data(config)
    
    # Make sure the data directory exists
    os.makedirs(config["data_directory"]["raw"], exist_ok=True)

    # Save the combined DataFrame to a CSV file
    save_data(combined_df, config)

def get_safe(url, params, retries=3, timeout=10):
    """Helper function to perform a GET request with retries."""
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt == retries - 1:
                raise

def get_raw_data(id, start_date, tenor, end_date, api_key_name, dir, frequency='d'):
    """Fetches data from the FRED API and saves it to a CSV file.
    Args:
        id (str): The series ID for the data to fetch.
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        tenor (str): The tenor for the data (e.g., '1MO', '3MO', etc.).
        end_date (str): The end date for the data in 'YYYY-MM-DD' format (default is today).
        api_key_name (str): The name of the environment variable containing the FRED API key.
        dir (str): The directory to save the CSV file.
        frequency (str): The frequency of the data ('d' for daily, 'm' for monthly, etc.).
    Returns:
        pd.DataFrame: A DataFrame containing the fetched data.
    """
    
    # Load the FRED API key from the environment variable
    fred_api_key = os.getenv(api_key_name)
    if not fred_api_key:
        raise ValueError(f"{api_key_name} environment variable is not set.")
    # Define the FRED API URL and parameters
    fred_url = 'https://api.stlouisfed.org/fred/series/observations?'
    params = {
        'api_key': fred_api_key,
        'file_type': 'json',
        'series_id': id,  # Example series ID for 10-Year Treasury
        'frequency': frequency,  # Daily frequency
        'observation_start': start_date,  # Start date for the data
        'observation_end': end_date  # End date for the data
    }
    # Fetch the data from FRED
    response = get_safe(fred_url, params)
    if response.status_code != 200:
        raise ValueError(f"Error fetching data from FRED API: {response.status_code} - {response.text}")
    # Parse the JSON response
    response = response.json()
    if 'observations' not in response:
        raise ValueError("No observations found in the response from FRED API.")
    # Convert the observations to a DataFrame and save it to a CSV file casting to numbers and NaN for blanks
    data_df = pd.DataFrame(response['observations'])
    data_df['value'] = pd.to_numeric(data_df['value'], errors='coerce')
    data_df['date'] = pd.to_datetime(data_df['date'], format='%Y-%m-%d')
    data_df = data_df[['date', 'value']]  # Keep only the date and value columns
    data_df = data_df.sort_values("date").drop_duplicates("date")
    # Save the DataFrame to a CSV file
    csv_file_path = os.path.join(dir, f"{id}_{tenor}.csv")
    data_df.to_csv(csv_file_path, index=False)
    return data_df 
    
def fetch_all_data(config):
    """Fetches data for all series IDs and combines them into a single DataFrame.
    Args:
        series_ids (dict): A dictionary mapping series IDs to their tenors.
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.
        frequency (str): The frequency of the data ('d' for daily, 'm' for monthly, etc.).
    Returns:
        pd.DataFrame: A DataFrame containing the combined data for all series IDs.
    """
    
    # Load configuration
    series_ids = config['tenors']
    start_date = config['start_date']
    end_date = config['end_date'] or datetime.today().strftime('%Y-%m-%d')
    frequency = config['frequency'].lower()
    api_key_name = config.get('api_key_env', 'FRED_API_KEY')
    dir = config["data_directory"]["raw"]
    
    # Create a DataFrame to hold the combined data with a date index
    date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
    combined_df = pd.DataFrame(date_range, columns=['date'])

    for series_id in series_ids.keys():
        try:
            data_df = get_raw_data(series_id, start_date, series_ids[series_id], end_date, api_key_name, dir, frequency)
            data_df.rename(columns={'value': series_ids[series_id]}, inplace=True)
            combined_df = pd.merge(combined_df, data_df, on='date', how='left')
            print(f"Data for {series_id} ({series_ids[series_id]}) fetched successfully.")
            print(data_df.head())  # Display the first few rows of the DataFrame
        except Exception as e:
            print(f"An error occurred while fetching data for {series_id}: {e}")
    return combined_df

def save_data(df, config):
    """Saves the DataFrame to a CSV file at the specified path.
    Args:
        df (pd.DataFrame): The DataFrame to save.
        path (str): The file path where the CSV should be saved.
    """
    dir = config["data_directory"]["raw"]
    csv_file_path = os.path.join(dir, "combined_data.csv")
    df.to_csv(csv_file_path, index=False)
    print(f"Data saved to {csv_file_path}")

if __name__ == "__main__":
    main()
