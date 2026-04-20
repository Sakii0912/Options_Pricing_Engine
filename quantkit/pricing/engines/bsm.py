"""Black-Scholes-Merton option pricing engine for European options"""

import math
from scipy.stats import norm
from ..core.instruments import Option, OptionStyle, OptionType, OptionPriceResult
from ..core.market import MarketData
import numpy as np

class BSMEngine:
    """
    Black-Scholes-Merton pricing engine for European options.
    Supports options with continuous dividend yield.
    Discrete dividends: uses spot-adjustment approximation for European options only.
    """

    @staticmethod
    def _calculate_d1_d2(S: float, K: float, r: float, q: float, sigma: float, T: float) -> tuple:
        """
        Calculate d1 and d2 for Black-Scholes formula.

        Args:
            S: Spot price (may be adjusted for discrete dividends)
            K: Strike price
            r: Risk-free rate
            q: Dividend yield (continuous only)
            sigma: Volatility
            T: Time to maturity

        Returns:
            Tuple of (d1, d2)
        """
        if T <= 0:
            return (0.0, 0.0)

        sqrt_T = math.sqrt(T)
        numerator = math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T
        d1 = numerator / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        return d1, d2

    @staticmethod
    def _price_call(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
        """
        Price a European call option.

        Formula: C = S*e^(-q*T)*N(d1) - K*e^(-r*T)*N(d2)
        """
        if T <= 0:
            return max(S - K, 0.0)

        d1, d2 = BSMEngine._calculate_d1_d2(S, K, r, q, sigma, T)

        call = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        return call

    @staticmethod
    def _price_put(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
        """
        Price a European put option.

        Formula: P = K*e^(-r*T)*N(-d2) - S*e^(-q*T)*N(-d1)
        """
        if T <= 0:
            return max(K - S, 0.0)

        d1, d2 = BSMEngine._calculate_d1_d2(S, K, r, q, sigma, T)

        put = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
        return put

    @staticmethod
    def price(option: Option, market: MarketData) -> float:
        """
        Price an option using Black-Scholes-Merton formula.

        Args:
            option: Option contract
            market: Market data

        Returns:
            Option price

        Raises:
            ValueError: If American option with discrete dividends (no closed-form solution)
        """
        S = market.spot
        K = option.strike
        T = option.maturity
        r = market.rate
        q = market.dividend_yield
        sigma = market.volatility

        # Check for unsupported combination
        if option.style == OptionStyle.AMERICAN and market.has_discrete_dividends(T):
            raise ValueError(
                "BSM engine does not support American options with discrete dividends. "
                "Use Binomial Tree engine instead."
            )

        # For European options with discrete dividends, use spot adjustment
        S_eff = S
        if market.has_discrete_dividends(T):
            pv_div = market.pv_discrete_dividends(T)
            S_eff = S - pv_div

        if option.option_type == OptionType.CALL:
            price =  BSMEngine._price_call(S_eff, K, r, q, sigma, T)
        else:
            price =  BSMEngine._price_put(S_eff, K, r, q, sigma, T)

        return OptionPriceResult(price=price, boundary_times=np.array([]), boundary_spots=np.array([]))
