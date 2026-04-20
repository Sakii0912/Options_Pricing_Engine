"""Binomial Tree option pricing engine for European and American options"""

import math
import numpy as np
from scipy.interpolate import interp1d
from ..core.instruments import Option, OptionStyle, OptionType, OptionPriceResult
from ..core.market import MarketData


class BinomialTreeEngine:
    """
    Binomial Tree pricing engine supporting both European and American options.
    Supports CRR (Cox-Ross-Rubinstein) and JR (Jarrow-Rudd) models.
    Handles continuous dividend yield and discrete dividend events.
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
        Calculate tree parameters (u, d, p, dt).

        Args:
            market: Market data
            T: Time to maturity

        Returns:
            Tuple of (u, d, p, dt) where p is risk-neutral probability
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

        # Risk-neutral probability with continuous dividend yield
        exp_rdt = math.exp((r - div_yield) * dt)
        p = (exp_rdt - d) / (u - d)

        # Validate probability is in (0, 1)
        if not (0 < p < 1):
            raise ValueError(
                f"Risk-neutral probability {p} out of bounds [0, 1]. "
                f"Try increasing steps or reducing volatility."
            )

        return u, d, p, dt

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

    def _apply_discrete_dividend(self, price_tree: np.ndarray, option: Option,
                                 market: MarketData, div_step: int, dt: float) -> np.ndarray:
        """
        Apply discrete dividend at a given tree step.

        At the ex-dividend step, each spot S becomes S - D (value just after dividend).
        We interpolate option values from before-dividend grid to after-dividend spots.

        Args:
            price_tree: Stock price tree
            option: Option contract
            market: Market data
            div_step: Tree step number where dividend occurs (0-indexed)
            dt: Time step

        Returns:
            Updated option tree after dividend adjustment
        """
        # Find dividend amount for this step
        div_amount = None
        for div in market.discrete_dividends:
            if abs(div.time - div_step * dt) < 1e-6:
                div_amount = div.amount
                break

        if div_amount is None:
            return None  # No adjustment needed

        # Get option tree at this step (before dividend)
        # We'll compute values at ex-dividend spots by interpolation
        # This is handled in backward induction; return dividend amount
        return div_amount

    def _backward_induction(self, price_tree: np.ndarray, option: Option,
                           market: MarketData, K: float, p: float, r: float, dt: float) -> float:
        """
        Perform backward induction to compute option price.
        Handles discrete dividends by spot adjustment at ex-dividend steps.

        Args:
            price_tree: Stock price tree
            option: Option contract
            market: Market data (including discrete dividends)
            K: Strike price
            p: Risk-neutral probability
            r: Risk-free rate
            dt: Time step

        Returns:
            Option price
        """
        option_tree = np.zeros((self.steps + 1, self.steps + 1))
        discount_factor = math.exp(-r * dt)
        is_call = option.option_type == OptionType.CALL
        is_american = option.style == OptionStyle.AMERICAN

        # Map dividend times to tree steps
        div_steps = {}
        for div in market.discrete_dividends:
            step_idx = round(div.time / dt)
            if 0 <= step_idx <= self.steps:
                div_steps[step_idx] = div.amount

        # Initialize payoffs at maturity
        for j in range(self.steps + 1):
            S_T = price_tree[self.steps][j]
            if is_call:
                option_tree[self.steps][j] = max(S_T - K, 0.0)
            else:
                option_tree[self.steps][j] = max(K - S_T, 0.0)


        boundary_times = []
        boundary_spots = []

        # Backward induction
        for i in range(self.steps - 1, -1, -1):
            # Check if there's a dividend at step i (ex-dividend step)
            has_div_at_i = i in div_steps
            div_amount = div_steps.get(i, 0.0)

            step_boundary_spot = None

            for j in range(i + 1):
                S_current = price_tree[i][j]

                if has_div_at_i:
                    # Ex-dividend step: apply dividend adjustment
                    # The stock price drops by dividend amount
                    S_ex_div = S_current - div_amount

                    # Interpolate option value at ex-dividend spot from next layer
                    # Build interpolation function from the nodes at i+1
                    spots_next = np.array([price_tree[i + 1][jj] for jj in range(i + 2)])
                    values_next = np.array([option_tree[i + 1][jj] for jj in range(i + 2)])

                    # Linear interpolation
                    interp_func = interp1d(spots_next, values_next, kind='linear',
                                          bounds_error=True, fill_value='extrapolate')
                    value_at_ex_div = interp_func(S_ex_div)

                    # Discount back
                    continuation = discount_factor * value_at_ex_div

                else:
                    # Standard backward induction (no dividend)
                    continuation = discount_factor * (
                        p * option_tree[i + 1][j + 1] +
                        (1.0 - p) * option_tree[i + 1][j]
                    )

                # For American options, check early exercise
                if is_american:
                    exercise = max(S_current - K, 0.0) if is_call else max(K - S_current, 0.0)
                    if exercise > continuation + 1e-9:
                        if is_call:
                            # For calls, find the MINIMUM stock price that triggers exercise
                            if step_boundary_spot is None or S_current < step_boundary_spot:
                                step_boundary_spot = S_current
                        else:
                            # For puts, find the MAXIMUM stock price that triggers exercise
                            if step_boundary_spot is None or S_current > step_boundary_spot:
                                step_boundary_spot = S_current
                    
                    option_tree[i][j] = max(continuation, exercise)
                else:
                    option_tree[i][j] = continuation

            if step_boundary_spot is not None:
                boundary_times.append(i * dt)
                boundary_spots.append(step_boundary_spot)

        return option_tree[0][0], np.array(boundary_times), np.array(boundary_spots)

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
        u, d, p, dt = self._calculate_tree_parameters(market, T)

        # Build price tree
        price_tree = self._build_price_tree(S, u, d)

        # Compute option price via backward induction (handles discrete dividends internally)
        option_price, b_times, b_spots = self._backward_induction(price_tree, option, market, K, p, r, dt)

        return OptionPriceResult(price=option_price, boundary_times=b_times, boundary_spots=b_spots)
