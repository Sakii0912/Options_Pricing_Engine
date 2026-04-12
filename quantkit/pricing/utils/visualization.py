"""Visualization utilities for boundaries and analysis"""

import matplotlib.pyplot as plt
from typing import Optional


import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from ..core.instruments import *

def plot_exercise_boundaries(
    instrument, # Using the Option dataclass defined earlier
    lsmc_res,   # The OptionPriceResult from LSMC
    benchmark_res: Optional['OptionPriceResult'] = None,
    title: str = "American Option Optimal Exercise Boundary"
):
    """
    Plots the optimal exercise boundary learned by LSMC, optionally comparing
    it against an exact benchmark (e.g., from a Binomial Tree or PDE).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    times = lsmc_res.boundary_times
    lsmc_boundary = lsmc_res.boundary_spots
    
    # 1. Plot the LSMC Boundary
    ax.plot(times, lsmc_boundary, label="LSMC Boundary", 
            linewidth=2, color="blue", marker=".", markersize=4)

    # 2. Plot the Benchmark Boundary (if provided)
    if benchmark_res is not None:
        ax.plot(benchmark_res.boundary_times, benchmark_res.boundary_spots, 
                label="Benchmark (Tree/DP) Exact Boundary", 
                linewidth=2, color="black", linestyle="--")

    # 3. Shade the Exercise and Continuation Zones
    # Matplotlib automatically ignores np.nan values when shading
    if instrument.option_type.value == "put":
        # For a put, you exercise when the stock price falls BELOW the boundary
        ax.fill_between(times, 0, lsmc_boundary, color="red", alpha=0.1, label="Exercise Zone")
        ax.fill_between(times, lsmc_boundary, ax.get_ylim()[1] * 1.5, color="green", alpha=0.1, label="Continuation Zone")
    else:
        # For a call, you exercise when the stock price rises ABOVE the boundary
        ax.fill_between(times, lsmc_boundary, ax.get_ylim()[1] * 1.5, color="red", alpha=0.1, label="Exercise Zone")
        ax.fill_between(times, 0, lsmc_boundary, color="green", alpha=0.1, label="Continuation Zone")

    # 4. Formatting
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time to Maturity (Years)", fontsize=12)
    ax.set_ylabel("Underlying Asset Price", fontsize=12)
    
    # Invert X-axis so T=0 is today, and T=Maturity is on the right
    ax.set_xlim(0, times[-1])
    
    # Set reasonable Y-limits based on the data
    valid_spots = lsmc_boundary[~np.isnan(lsmc_boundary)]
    if len(valid_spots) > 0:
        y_min = max(0, valid_spots.min() * 0.8)
        y_max = valid_spots.max() * 1.2
        ax.set_ylim(y_min, y_max)

    # Horizontal line for the Strike Price
    ax.axhline(instrument.strike, color='grey', linestyle=':', label=f"Strike (K={instrument.strike})")

    ax.legend(loc="best", frameon=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()


def plot_comparison(results: dict, title: str = "Pricing Comparison") -> None:
    """
    Compare pricing results across engines.
    
    Args:
        results: Dictionary with pricing results
        title: Plot title
    """
    raise NotImplementedError("To be implemented")
