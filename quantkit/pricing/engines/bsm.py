"""Black-Scholes-Merton option pricing engine"""

import math
from scipy.stats import norm

from ..core.instruments import Option, OptionType
from ..core.market import MarketData


class BSMEngine:
    """
    Black-Scholes-Merton pricing engine for European options.
    """

    @staticmethod
    def _calculate_d1_d2(S: float, K: float, r: float, sigma: float, T: float) -> tuple:
        """
        Calculate d1 and d2 for Black-Scholes formula.

        Args:
            S: Spot price
            K: Strike price
            r: Risk-free rate
            sigma: Volatility
            T: Time to maturity

        Returns:
            Tuple of (d1, d2)
        """
        if T <= 0:
            # At maturity, derivatives approach intrinsic value
            return (math.inf, math.inf) if S > K else (-math.inf, -math.inf)

        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        return d1, d2

    @staticmethod
    def _european_call(S: float, K: float, r: float, sigma: float, T: float) -> float:
        """
        Calculate European call option price using Black-Scholes formula.

        Formula: C = S*N(d1) - K*e^(-r*T)*N(d2)

        Args:
            S: Spot price
            K: Strike price
            r: Risk-free rate
            sigma: Volatility
            T: Time to maturity

        Returns:
            Call option price
        """
        if T <= 0:
            # At maturity, call is worth its intrinsic value
            return max(S - K, 0.0)

        d1, d2 = BSMEngine._calculate_d1_d2(S, K, r, sigma, T)

        call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

        return call_price

    @staticmethod
    def _european_put(S: float, K: float, r: float, sigma: float, T: float) -> float:
        """
        Calculate European put option price using Black-Scholes formula.

        Formula: P = K*e^(-r*T)*N(-d2) - S*N(-d1)

        Args:
            S: Spot price
            K: Strike price
            r: Risk-free rate
            sigma: Volatility
            T: Time to maturity

        Returns:
            Put option price
        """
        if T <= 0:
            # At maturity, put is worth its intrinsic value
            return max(K - S, 0.0)

        d1, d2 = BSMEngine._calculate_d1_d2(S, K, r, sigma, T)

        put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return put_price

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
            ValueError: If option is not European style
        """
        if option.option_type == OptionType.CALL:
            return BSMEngine._european_call(
                S=market.spot,
                K=option.strike,
                r=market.rate,
                sigma=market.volatility,
                T=option.maturity
            )
        elif option.option_type == OptionType.PUT:
            return BSMEngine._european_put(
                S=market.spot,
                K=option.strike,
                r=market.rate,
                sigma=market.volatility,
                T=option.maturity
            )
        else:
            raise ValueError(f"Unknown option type: {option.option_type}")
