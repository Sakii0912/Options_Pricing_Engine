"""Exercise boundary extraction and export utilities"""

import numpy as np
from typing import Tuple, Dict


def extract_boundary(prices: np.ndarray, strikes: np.ndarray, times: np.ndarray) -> Dict:
    """
    Extract exercise boundary from price grid.
    
    Args:
        prices: Price grid
        strikes: Strike prices
        times: Time points
        
    Returns:
        Dictionary with boundary data
    """
    raise NotImplementedError("To be implemented")


def export_boundary(boundary_data: Dict, filepath: str) -> None:
    """
    Export boundary data to file.
    
    Args:
        boundary_data: Boundary dictionary
        filepath: Output file path
    """
    raise NotImplementedError("To be implemented")
