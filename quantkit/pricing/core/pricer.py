"""Pricing router - delegates to appropriate engine"""

from .instruments import Option, OptionStyle
from .market import MarketData


class Pricer:
    """
    Main pricing interface that routes to appropriate engine based on option style.
    """
    
    @staticmethod
    def price(option: Option, market: MarketData, engine: str = "auto", **kwargs) -> float:
        """
        Price an option.
        
        Args:
            option: Option contract
            market: Market data
            engine: Engine to use ("auto", "bsm", "binomial", "lsmc")
            **kwargs: Engine-specific parameters
            
        Returns:
            Option price
        """
        raise NotImplementedError("To be implemented")
