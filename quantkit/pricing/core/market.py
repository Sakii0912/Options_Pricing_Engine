"""Market data specification"""

from dataclasses import dataclass, field
from typing import List
import numpy as np

@dataclass
class DividendEvent:
    """
    A discrete dividend payment.

    Attributes:
        time: Time of dividend payment in years (ex-dividend date)
        amount: Cash dividend amount per share
    """
    time: float
    amount: float
@dataclass
class MarketData:
    """
    Market parameters for option pricing.

    Attributes:
        spot: Current spot price (S0)
        rate: Risk-free interest rate (r)
        volatility: Asset volatility (sigma)
        dividend_yield: Dividend yield (q), optional
        discrete_dividends: List of discrete dividend events, optional
    """
    spot: float
    rate: float
    volatility: float
    dividend_yield: float = 0.0
    discrete_dividends: List[DividendEvent] = field(default_factory=list)

    def has_discrete_dividends(self, T: float) -> bool:
        """Check if any discrete dividends occur before maturity T."""
        return any(div.time < T for div in self.discrete_dividends)

    def pv_discrete_dividends(self, T: float) -> float:
        """
        Compute present value of discrete dividends occurring before maturity T.

        Returns:
            Sum of discounted dividend amounts.
        """
        pv = 0.0
        for div in self.discrete_dividends:
            if div.time < T:
                pv += div.amount * np.exp(-self.rate * div.time)
        return pv

    def dividends_in_range(self, t_start: float, t_end: float) -> List[DividendEvent]:
        """Get all discrete dividends in the time range [t_start, t_end)."""
        return [div for div in self.discrete_dividends if t_start <= div.time < t_end]
