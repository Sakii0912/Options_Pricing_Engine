"""Option instrument definitions"""

from enum import Enum
from dataclasses import dataclass
import numpy as np


class OptionType(Enum):
    """Option type: Call or Put"""
    CALL = "call"
    PUT = "put"


class OptionStyle(Enum):
    """Option exercise style: European or American"""
    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass
class Option:
    """
    Option contract specification.
    
    Attributes:
        strike: Strike price (K)
        maturity: Time to maturity in years (T)
        option_type: OptionType.CALL or OptionType.PUT
        style: OptionStyle.EUROPEAN or OptionStyle.AMERICAN
    """
    strike: float
    maturity: float
    option_type: OptionType
    style: OptionStyle

@dataclass
class OptionPriceResult:
    """Standardized output containing price and the extracted boundary."""
    price: float
    boundary_times: np.ndarray
    boundary_spots: np.ndarray

class BasketType(Enum):
    """Aggregation type for basket options"""
    AVERAGE = "average"
    MAX = "max"
    MIN = "min"

@dataclass
class BasketOption(Option):
    """
    Basket Option contract specification.
    
    Attributes:
        basket_type: How the underlying assets are aggregated
        weights: Optional weights for the basket components (defaults to equal weights if None)
    """
    basket_type: BasketType = BasketType.AVERAGE
    weights: np.ndarray = None
