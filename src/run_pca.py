import pandas as pd
from pca import run_pca

def main():
    """Main function to run PCA on the cleaned data and save the results."""
    df = pd.read_csv("data/processed/cleaned_data_diffs.csv", parse_dates=['date']).set_index('date')

    factors, loadings, explained = run_pca(df)

    factors.to_csv("data/processed/pca_factors.csv")
    loadings.index.name = "tenor"
    loadings.to_csv("data/processed/pca_loadings.csv")

    print("PCA complete. Variance explained:", explained)
    
if __name__ == "__main__":
    main()