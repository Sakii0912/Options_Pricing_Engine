"""Binomial Tree option pricing engine for European and American options"""

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
        Initialize binomial tree engine.

        Args:
            steps: Number of tree steps
            model: "crr" (Cox-Ross-Rubinstein) or "jr" (Jarrow-Rudd)
        """
        self.steps = steps
        self.model = model.lower()
        if self.model not in ["crr", "jr"]:
            raise ValueError(f"Unknown model: {model}. Use 'crr' or 'jr'")

    def _calculate_tree_parameters(self, market: MarketData, T: float) -> tuple:
        """
        Calculate tree parameters (u, d, q, dt).

        Args:
            market: Market data
            T: Time to maturity

        Returns:
            Tuple of (u, d, q, dt) where q is risk-neutral probability
        """
        dt = T / self.steps
        sqrt_dt = math.sqrt(dt)
        sigma = market.volatility
        r = market.rate
        div_yield = market.dividend_yield

        if self.model == "crr":
            # Cox-Ross-Rubinstein model
            u = math.exp(sigma * sqrt_dt)
            d = 1.0 / u
        else:  # JR model
            # Jarrow-Rudd model
            drift = (r - div_yield - 0.5 * sigma ** 2) * dt
            u = math.exp(drift + sigma * sqrt_dt)
            d = math.exp(drift - sigma * sqrt_dt)

        # Risk-neutral probability
        exp_rdt = math.exp(r * dt)
        q = (exp_rdt - d) / (u - d)

        return u, d, q, dt

    def _build_price_tree(self, S0: float, u: float, d: float) -> np.ndarray:
        """
        Build the stock price tree.

        Args:
            S0: Initial stock price
            u: Up factor
            d: Down factor

        Returns:
            Stock price tree as 2D array
        """
        tree = np.zeros((self.steps + 1, self.steps + 1))

        for i in range(self.steps + 1):
            for j in range(i + 1):
                tree[i][j] = S0 * (u ** j) * (d ** (i - j))

        return tree

    def _backward_induction(self, price_tree: np.ndarray, option: Option,
                           K: float, q: float, r: float, dt: float) -> float:
        """
        Perform backward induction to compute option price.

        Args:
            price_tree: Stock price tree
            option: Option contract
            K: Strike price
            q: Risk-neutral probability
            r: Risk-free rate
            dt: Time step

        Returns:
            Option price
        """
        option_tree = np.zeros((self.steps + 1, self.steps + 1))
        discount_factor = math.exp(-r * dt)
        is_call = option.option_type == OptionType.CALL
        is_american = option.style == OptionStyle.AMERICAN

        # Initialize payoffs at maturity
        for j in range(self.steps + 1):
            S_T = price_tree[self.steps][j]
            if is_call:
                option_tree[self.steps][j] = max(S_T - K, 0.0)
            else:
                option_tree[self.steps][j] = max(K - S_T, 0.0)

        # Backward induction
        for i in range(self.steps - 1, -1, -1):
            for j in range(i + 1):
                # Discounted expected value (continuation value)
                continuation = discount_factor * (
                    q * option_tree[i + 1][j + 1] +
                    (1.0 - q) * option_tree[i + 1][j]
                )

                # For American options, check early exercise
                if is_american:
                    S_current = price_tree[i][j]
                    if is_call:
                        exercise = max(S_current - K, 0.0)
                    else:
                        exercise = max(K - S_current, 0.0)
                    option_tree[i][j] = max(continuation, exercise)
                else:
                    option_tree[i][j] = continuation

        return option_tree[0][0]

    def price(self, option: Option, market: MarketData) -> float:
        """
        Price an option using binomial tree method.

        Args:
            option: Option contract
            market: Market data

        Returns:
            Option price
        """
        S = market.spot
        K = option.strike
        T = option.maturity
        r = market.rate

        if T <= 0:
            # At maturity, return intrinsic value
            if option.option_type == OptionType.CALL:
                return max(S - K, 0.0)
            else:
                return max(K - S, 0.0)

        # Calculate tree parameters
        u, d, q, dt = self._calculate_tree_parameters(market, T)

        # Build price tree
        price_tree = self._build_price_tree(S, u, d)

        # Compute option price via backward induction
        option_price = self._backward_induction(price_tree, option, K, q, r, dt)

        return option_price
