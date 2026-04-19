"""Main pricing router - delegates to appropriate engine"""

from .instruments import Option, OptionStyle, OptionType
from .market import MarketData
from ..engines.bsm import BSMEngine
from ..engines.binomial_tree import BinomialTreeEngine


class Pricer:
    """
    Main pricing interface that routes options to the appropriate engine.

    Routing Logic:
    - European, no discrete dividends -> BSM (default)
    - European, with discrete dividends -> Binomial Tree (spot-adjust BSM is approximate)
    - American, no discrete dividends, Call with q=0 -> BSM (early exercise suboptimal)
    - American with any discrete dividends or other -> Binomial Tree
    """

    @staticmethod
    def price(option: Option, market: MarketData, engine: str = "auto",
              steps: int = 100, model: str = "crr") -> float:
        """
        Price an option using the appropriate engine.

        Args:
            option: Option contract
            market: Market data (including dividend_yield and discrete_dividends)
            engine: Engine preference:
                - "auto": Uses BSM for European (no discrete div), Binomial for others (default)
                - "bsm": Forces Black-Scholes-Merton (fails if unsupported combination)
                - "binomial": Forces Binomial Tree method
            steps: Number of tree steps for binomial tree (default: 100)
            model: Binomial model ("crr" or "jr", default: "crr")

        Returns:
            Option price

        Raises:
            ValueError: If unsupported option configuration or engine
        """
        engine = engine.lower()

        # Validate engine choice
        if engine not in ["auto", "bsm", "binomial"]:
            raise ValueError(f"Unknown engine: {engine}. Use 'auto', 'bsm', or 'binomial'")

        # European options
        if option.style == OptionStyle.EUROPEAN:
            if engine == "bsm":
                return BSMEngine.price(option, market)
            elif engine == "binomial":
                tree_engine = BinomialTreeEngine(steps=steps, model=model)
                return tree_engine.price(option, market)
            else:  # auto
                # Use BSM for European if no discrete dividends, else use Binomial
                if market.has_discrete_dividends(option.maturity):
                    tree_engine = BinomialTreeEngine(steps=steps, model=model)
                    return tree_engine.price(option, market)
                else:
                    return BSMEngine.price(option, market)

        # American options
        elif option.style == OptionStyle.AMERICAN:
            # American call with no dividend and no discrete dividends -> use BSM
            # (early exercise is never optimal for non-dividend-paying asset)
            if (option.option_type == OptionType.CALL and
                market.dividend_yield == 0.0 and
                not market.has_discrete_dividends(option.maturity)):
                if engine == "binomial":
                    tree_engine = BinomialTreeEngine(steps=steps, model=model)
                    return tree_engine.price(option, market)
                else:
                    # Default to BSM for this special case
                    return BSMEngine.price(option, market)

            # All other American cases -> Binomial Tree
            tree_engine = BinomialTreeEngine(steps=steps, model=model)
            return tree_engine.price(option, market)

        else:
            raise ValueError(f"Unsupported option style: {option.style}")
