import pandas as pd
import os

def main():
    """Main function to clean the combined data CSV file."""
    cleaned_data = clean_data()
    # Save the cleaned data to a new CSV file
    cleaned_csv_path = os.path.join('data', 'processed', 'cleaned_data.csv')
    os.makedirs(os.path.dirname(cleaned_csv_path), exist_ok=True)
    cleaned_data.to_csv(cleaned_csv_path, index=False)
    print(f"Cleaned data saved to {cleaned_csv_path}")

def clean_data(path='data/raw/combined_data.csv'):
    """Reads data from the combined CSV file, ensure the columns being read are expected, and cleans the data.
    Args:
        path (str): The path to the CSV file containing the combined data.
    Returns:
        pd.DataFrame: A cleaned DataFrame with the expected columns.
    """
    data = pd.read_csv(path, parse_dates=['date'])
    expected_columns = ['date', '1MO', '3MO', '6MO', '1Y', '2Y', '3Y', '5Y', '7Y',
                        '10Y', '20Y', '30Y']
    if not all(col in data.columns for col in expected_columns):
        raise ValueError(f"Data does not contain all expected columns: {expected_columns}")
    # Ensure the date column is in datetime format
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    # Drop rows with NaN values in the data columns
    data.dropna(subset=expected_columns[1:], inplace=True)
    # Reset index for the cleaned DataFrame
    data.reset_index(drop=True, inplace=True)
    return data

if __name__ == "__main__":
    main()
