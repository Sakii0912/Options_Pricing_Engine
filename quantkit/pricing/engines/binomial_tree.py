"""Binomial Tree option pricing engine"""

import math
import numpy as np
from ..core.instruments import Option, OptionStyle, OptionType
from ..core.market import MarketData

class BinomialTreeEngine:
    """
    Binomial Tree pricing engine supporting both European and American options.
    Supports CRR (Cox-Ross-Rubinstein) and JR (Jarrow-Rudd) models.
    """
    
    def __init__(self, steps: int = 100, model: str = "crr"):
        """
        Initialize the binomial tree engine.
        steps: Number of time steps in the tree
        model: Tree model to use ("crr" or "jr")
        """
        self.steps = steps
        self.model = model
    
    def price(self, option: Option, market: MarketData) -> float:
        """
        Price the option using a binomial tree.
        option: Option to price
        market: Market data for pricing

        """
        S = market.spot
        K = option.strike
        T = option.maturity
        r = market.rate
        q = market.dividend_yield
        sigma = market.volatility
        
        if T <= 0:
             return max(0.0, S - K) if option.option_type == OptionType.CALL else max(0.0, K - S)
             
        dt = T / self.steps
        
        # Calculate u, d, and p based on the chosen model (defaulting to CRR)
        if self.model.lower() == "crr":
            u = math.exp(sigma * math.sqrt(dt))
            d = math.exp(-sigma * math.sqrt(dt))
            p = (math.exp((r - q) * dt) - d) / (u - d)
        elif self.model.lower() == "jr":
            # Jarrow-Rudd parameterization
            u = math.exp((r - q - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt))
            d = math.exp((r - q - 0.5 * sigma**2) * dt - sigma * math.sqrt(dt))
            p = 0.5
        else:
            raise ValueError(f"Unsupported tree model: {self.model}")
            
        discount = math.exp(-r * dt)
        
        # Initialize terminal asset prices
        asset_prices = S * (u ** np.arange(self.steps, -1, -1)) * (d ** np.arange(0, self.steps + 1))
        
        # Initialize terminal payoffs
        if option.option_type == OptionType.CALL:
            values = np.maximum(0, asset_prices - K)
        else:
            values = np.maximum(0, K - asset_prices)
            
        # Backward induction
        for i in range(self.steps - 1, -1, -1):
            # Continuation value
            values = discount * (p * values[:-1] + (1 - p) * values[1:])
            
            # Early exercise check for American options
            if option.style == OptionStyle.AMERICAN:
                current_asset_prices = S * (u ** np.arange(i, -1, -1)) * (d ** np.arange(0, i + 1))
                if option.option_type == OptionType.CALL:
                    exercise = np.maximum(0, current_asset_prices - K)
                else:
                    exercise = np.maximum(0, K - current_asset_prices)
                values = np.maximum(values, exercise)
                
        return values[0]