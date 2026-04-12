"""Pricing router - delegates to appropriate engine"""

from .instruments import Option, OptionStyle, OptionType
from .market import MarketData
from ..engines.bsm import BSMEngine
from ..engines.binomial_tree import BinomialTreeEngine


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
        if option.style == OptionStyle.EUROPEAN:
            return BSMEngine.price(option, market)
        
        elif option.style == OptionStyle.AMERICAN:

            if option.option_type == OptionType.CALL and market.dividend_yield == 0.0:
                # For non-dividend paying stocks, American call = European call
                return BSMEngine.price(option, market)

            else:
                if engine.lower() == "binomial":
                    return BinomialTreeEngine(**kwargs).price(option, market)
                # else:
                #     # Default to binomial tree for American options
                #     return BinomialTreeEngine(**kwargs).price(option, market)