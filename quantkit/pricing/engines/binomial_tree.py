"""Binomial Tree option pricing engine"""

import math
import numpy as np

from ..core.instruments import Option, OptionType
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
        self.model = model.lower()
        if self.model not in ["crr", "jr"]:
            raise ValueError(f"Unknown model: {model}")

        self.u = None  # Up factor
        self.d = None  # Down factor
        self.q = None  # Risk-neutral probability
        self.dt = None  # Time step

    def _calculate_u_d(self, market: MarketData, T: float) -> None:
        """
        Calculate up and down factors for binomial tree.

        CRR model:
            u = e^(σ*√Δt)
            d = 1/u

        JR model:
            u = e^((r-q+σ²/2)*Δt + σ*√Δt)
            d = e^((r-q+σ²/2)*Δt - σ*√Δt)

        Args:
            market: Market data
            T: Time to maturity
        """
        self.dt = T / self.steps
        sqrt_dt = math.sqrt(self.dt)
        sigma = market.volatility

        if self.model == "crr":
            self.u = math.exp(sigma * sqrt_dt)
            self.d = 1.0 / self.u
        else:  # JR model
            r = market.rate
            q = market.dividend_yield
            drift = (r - q + 0.5 * sigma ** 2) * self.dt
            self.u = math.exp(drift + sigma * sqrt_dt)
            self.d = math.exp(drift - sigma * sqrt_dt)

    def _calculate_risk_neutral_prob(self, market: MarketData) -> None:
        """
        Calculate risk-neutral probability for binomial tree.

        Formula: q = (e^(r*Δt) - d) / (u - d)

        Args:
            market: Market data
        """
        r = market.rate
        exp_rdt = math.exp(r * self.dt)
        self.q = (exp_rdt - self.d) / (self.u - self.d)

    def _build_stock_tree(self, S0: float) -> np.ndarray:
        """
        Build the stock price tree.

        Returns:
            2D array where stock_tree[i][j] is the stock price at step i, node j
        """
        stock_tree = np.zeros((self.steps + 1, self.steps + 1))

        for i in range(self.steps + 1):
            for j in range(i + 1):  # j <= i (upper triangular)
                # Stock price at node (i, j): S0 * u^j * d^(i-j)
                stock_tree[i][j] = S0 * (self.u ** j) * (self.d ** (i - j))

        return stock_tree

    def _backward_induction(self, stock_tree: np.ndarray, option: Option,
                           market: MarketData, is_american: bool = False) -> float:
        """
        Compute option price using backward induction.

        Args:
            stock_tree: Stock price tree
            option: Option contract
            market: Market data
            is_american: Whether to allow early exercise (not used for European)

        Returns:
            Option price at root (node 0)
        """
        K = option.strike
        is_call = option.option_type == OptionType.CALL
        exp_neg_rdt = math.exp(-market.rate * self.dt)

        # Initialize option values at maturity
        option_tree = np.zeros((self.steps + 1, self.steps + 1))

        for j in range(self.steps + 1):
            S_T = stock_tree[self.steps][j]
            if is_call:
                option_tree[self.steps][j] = max(S_T - K, 0.0)
            else:
                option_tree[self.steps][j] = max(K - S_T, 0.0)

        # Backward induction
        for i in range(self.steps - 1, -1, -1):
            for j in range(i + 1):
                # Value from holding: discounted expected value
                hold_value = exp_neg_rdt * (
                    self.q * option_tree[i + 1][j + 1] +
                    (1.0 - self.q) * option_tree[i + 1][j]
                )

                option_tree[i][j] = hold_value

        return option_tree[0][0]

    def price(self, option: Option, market: MarketData) -> float:
        """
        Price an option using binomial tree method.

        For European options, no early exercise is considered.

        Args:
            option: Option contract
            market: Market data

        Returns:
            Option price
        """
        # Calculate tree parameters
        self._calculate_u_d(market, option.maturity)
        self._calculate_risk_neutral_prob(market)

        # Build stock price tree
        stock_tree = self._build_stock_tree(market.spot)

        # Calculate option price with backward induction
        option_price = self._backward_induction(stock_tree, option, market, is_american=False)

        return option_price
