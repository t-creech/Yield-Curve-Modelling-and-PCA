import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from key_rate_duration import KEY_RATE_TENORS

def plot_distribution(data, column, bins=50, title=None):
    """Plots the distribution of a specified column in the data and returns the figure."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data[column], bins=bins, alpha=0.7, edgecolor="black")
    ax.set_title(title or f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    return fig


def plot_krd_bar(krd_dict, title="Key Rate Duration"):
    """Plots a bar chart of a single KRD vector."""
    tenors = list(krd_dict.keys())
    values = list(krd_dict.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(tenors, values, color="steelblue")
    ax.set_title(title)
    ax.set_ylabel("Key Rate Duration")
    ax.grid(axis="y", alpha=0.3)
    return fig


def plot_krd_fan(krd_df, title="KRD Fan Chart"):
    """Plots a fan chart of multiple KRD vectors."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for col in krd_df.columns:
        ax.plot(krd_df.index, krd_df[col], alpha=0.1, color="steelblue")

    # plot the average KRD in bold
    ax.plot(krd_df.mean(axis=0), color="black", linewidth=2, label="Mean KRD")

    ax.set_title(title)
    ax.set_ylabel("Key Rate Duration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_yield_curves(curves, title="Yield Curve Scenarios"):
    """
    curves: dict {label: dataframe of tenors}
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, df in curves.items():
        ax.plot(df.columns, df.iloc[0], label=label)

    ax.set_title(title)
    ax.set_ylabel("Yield (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_term_premium_shifts(base_curve, shocked_curve, title="Term Premium Shift"):
    """Plots the base and shocked yield curves along with the shift between them."""
    
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(base_curve.columns, base_curve.iloc[0], label="Base", linewidth=2)
    ax.plot(shocked_curve.columns, shocked_curve.iloc[0], label="Shock", linestyle="--")

    shift = shocked_curve.iloc[0] - base_curve.iloc[0]
    ax.bar(base_curve.columns, shift, alpha=0.3, label="Shift")

    ax.set_title(title)
    ax.set_ylabel("Yield (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig
