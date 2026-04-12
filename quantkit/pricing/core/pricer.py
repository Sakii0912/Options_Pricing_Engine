"""Main pricing router - delegates to appropriate engine"""

from .instruments import Option, OptionStyle, OptionType
from .market import MarketData
from ..engines.bsm import BSMEngine
from ..engines.binomial_tree import BinomialTreeEngine


class Pricer:
    """
    Main pricing interface that routes options to the appropriate engine.

    Routing Logic:
    - European (Any) -> BSM (default) or Binomial Tree (if specified)
    - American (No Dividend) Call -> BSM (early exercise is suboptimal)
    - American Put -> Binomial Tree
    - American Call (with Dividend) -> Binomial Tree
    """

    @staticmethod
    def price(option: Option, market: MarketData, engine: str = "auto",
              steps: int = 100, model: str = "crr") -> float:
        """
        Price an option using the appropriate engine.

        Args:
            option: Option contract
            market: Market data
            engine: Engine preference:
                - "auto": Uses BSM for European, Binomial for American (default)
                - "bsm": Forces Black-Scholes-Merton
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
                # Default to BSM for European options
                return BSMEngine.price(option, market)

        # American options
        elif option.style == OptionStyle.AMERICAN:
            # American call with no dividend -> use BSM (early exercise is suboptimal)
            if option.option_type == OptionType.CALL and market.dividend_yield == 0.0:
                if engine == "binomial":
                    # User explicitly requested binomial
                    tree_engine = BinomialTreeEngine(steps=steps, model=model)
                    return tree_engine.price(option, market)
                else:
                    # Default to BSM
                    return BSMEngine.price(option, market)

            # American put or American call with dividend -> use Binomial Tree
            tree_engine = BinomialTreeEngine(steps=steps, model=model)
            return tree_engine.price(option, market)

        else:
            raise ValueError(f"Unsupported option style: {option.style}")
