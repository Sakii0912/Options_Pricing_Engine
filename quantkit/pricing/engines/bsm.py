"""Black-Scholes-Merton option pricing engine"""

import math
from scipy.stats import norm
from ..core.instruments import Option, OptionType
from ..core.market import MarketData

class BSMEngine:
    """Black-Scholes-Merton pricing engine for European options."""
    
    @staticmethod
    def price(option: Option, market: MarketData) -> float:
        S = market.spot
        K = option.strike
        T = option.maturity
        r = market.rate
        q = market.dividend_yield
        sigma = market.volatility
        
        if T <= 0:
            if option.option_type == OptionType.CALL:
                return max(0.0, S - K)
            return max(0.0, K - S)
            
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        if option.option_type == OptionType.CALL:
            return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:
            return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)