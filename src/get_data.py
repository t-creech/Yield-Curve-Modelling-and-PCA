## Script to pull data from FRED and save it to a CSV file in the data directory.
import os
import pandas as pd
import requests
from datetime import datetime

def main():
    series_ids = {
        'DGS1MO': '1MO',
        'DGS3MO': '3MO',
        'DGS6MO': '6MO',
        'DGS1': '1Y',
        'DGS2': '2Y',
        'DGS3': '3Y',
        'DGS5': '5Y',
        'DGS7': '7Y',
        'DGS10': '10Y',
        'DGS20': '20Y',
        'DGS30': '30Y'}

    start_date = os.getenv('START_DATE', '2000-01-01')
    end_date = os.getenv('END_DATE', datetime.today().strftime('%Y-%m-%d'))
    # Create a DataFrame to hold the combined data with a date index
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    combined_df = pd.DataFrame(date_range, columns=['date'])

    for series_id in series_ids.keys():
        try:
            data_df = get_raw_data(series_id, start_date, series_ids[series_id], end_date)
            data_df.rename(columns={'value': series_ids[series_id]}, inplace=True)
            combined_df = pd.merge(combined_df, data_df, on='date', how='left')
            print(f"Data for {series_id} ({series_ids[series_id]}) fetched successfully.")
            print(data_df.head())  # Display the first few rows of the DataFrame
        except Exception as e:
            print(f"An error occurred while fetching data for {series_id}: {e}")

    # Save the combined DataFrame to a CSV file
    combined_csv_path = os.path.join('data', 'raw', 'combined_data.csv')
    os.makedirs(os.path.dirname(combined_csv_path), exist_ok=True)
    combined_df.to_csv(combined_csv_path, index=False)

def get_raw_data(id, start_date, path_add, end_date, frequency='d'):
    """Fetches data from the FRED API and saves it to a CSV file.
    Args:
        id (str): The series ID for the data to fetch.
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        path_add (str): Additional path to save the data.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format (default is today).
        frequency (str): The frequency of the data ('d' for daily, 'm' for monthly, etc.).
    Returns:
        pd.DataFrame: A DataFrame containing the fetched data.
    """
    # Ensure the data directory exists
    data_dir = os.path.join('data','raw')
    os.makedirs(data_dir, exist_ok=True)

    # Load the FRED API key from the environment variable
    fred_api_key = os.getenv('FRED_API_KEY')
    if not fred_api_key:
        raise ValueError("FRED_API_KEY environment variable is not set.")
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
    response = requests.get(fred_url, params=params, timeout=20)
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
    # Save the DataFrame to a CSV file
    csv_file_path = os.path.join(data_dir, f"{id}_{path_add}.csv")
    data_df.to_csv(csv_file_path, index=False)
    return data_df 

if __name__ == "__main__":
    main()
