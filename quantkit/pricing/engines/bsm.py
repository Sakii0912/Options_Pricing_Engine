"""Black-Scholes-Merton option pricing engine"""

from ..core.instruments import Option
from ..core.market import MarketData


class BSMEngine:
    """
    Black-Scholes-Merton pricing engine for European options.
    """
    
    @staticmethod
    def price(option: Option, market: MarketData) -> float:
        """
        Price an option using Black-Scholes-Merton formula.
        
        Args:
            option: Option contract
            market: Market data
            
        Returns:
            Option price
        """
        raise NotImplementedError("To be implemented")
