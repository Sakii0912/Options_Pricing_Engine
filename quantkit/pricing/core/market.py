"""Market data specification"""

from dataclasses import dataclass


@dataclass
class MarketData:
    """
    Market parameters for option pricing.
    
    Attributes:
        spot: Current spot price (S0)
        rate: Risk-free interest rate (r)
        volatility: Asset volatility (sigma)
        dividend_yield: Dividend yield (q), optional
    """
    spot: float
    rate: float
    volatility: float
    dividend_yield: float = 0.0
