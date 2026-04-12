"""Binomial Tree option pricing engine"""

from ..core.instruments import Option
from ..core.market import MarketData


class BinomialTreeEngine:
    """
    Binomial Tree pricing engine supporting both European and American options.
    Supports CRR (Cox-Ross-Rubinstein) and JR (Jarrow-Rudd) models.
    """
    
    def __init__(self, steps: int = 100, model: str = "crr"):
        """
        Initialize binomial tree engine.
        
        Args:
            steps: Number of tree steps
            model: "crr" or "jr"
        """
        self.steps = steps
        self.model = model
    
    def price(self, option: Option, market: MarketData) -> float:
        """
        Price an option using binomial tree method.
        
        Args:
            option: Option contract
            market: Market data
            
        Returns:
            Option price
        """
        raise NotImplementedError("To be implemented")
