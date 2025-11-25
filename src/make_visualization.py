import os
import pandas as pd
from config_loader import load_config
from monte_carlo_risk import compute_mc_analytics
from visualization import (
    plot_distribution,
    plot_krd_bar,
    plot_krd_fan,
    plot_yield_curves,
    plot_term_premium_shifts
)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    """Main function to generate visualizations from Monte Carlo risk analytics."""
    
    config = load_config()
    figures_dir = config["data_directory"]["figures"]
    reports_dir = config["data_directory"]["reports"]

    ensure_dir(figures_dir)
    ensure_dir(reports_dir)

    date = config["monte_carlo"]["evaluation"]
    print(f"Using MC evaluation date: {date}")
    
    mc_results_path = os.path.join(reports_dir, "simulated_yield_curve_analytics.csv")
    mc_df = pd.read_csv(mc_results_path)
    
    
    # MC Distribution Plots
    print("Generating MC distribution plots...")

    price_hist = plot_distribution(mc_df, "price", title="MC Price Distribution")
    mod_dur_hist = plot_distribution(mc_df, "modified_duration", title="MC Modified Duration Distribution")
    conv_hist = plot_distribution(mc_df, "convexity", title="MC Convexity Distribution")
    
    # Save distribution plots
    price_hist_path = os.path.join(figures_dir, "mc_price_distribution.png")
    mod_dur_hist_path = os.path.join(figures_dir, "mc_modified_duration_distribution.png")
    conv_hist_path = os.path.join(figures_dir, "mc_convexity_distribution.png")
    price_hist.savefig(price_hist_path)
    mod_dur_hist.savefig(mod_dur_hist_path)
    conv_hist.savefig(conv_hist_path)

    # KRD Fan Chart
    print("Generating KRD fan chart...")

    krd_cols = [c for c in mc_df.columns if c.startswith("krd_")]
    krd_df = mc_df[krd_cols]
    krd_fan = plot_krd_fan(krd_df, title="Monte Carlo Key Rate Duration Fan Chart")
    
    krd_fan_path = os.path.join(figures_dir, "mc_krd_fan_chart.png")
    krd_fan.savefig(krd_fan_path)

    # Single KRD Vector Bar Chart
    print("Plotting single KRD vector example...")
    first_krd = mc_df.iloc[0][krd_cols].to_dict()
    krd_vec_plot = plot_krd_bar(first_krd, title="Single-Path KRD")
    
    krd_vec_path = os.path.join(figures_dir, "krd_vector_bar_chart.png")
    krd_vec_plot.savefig(krd_vec_path)
    print("Visualization complete!")


if __name__ == "__main__":
    main()