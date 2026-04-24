"""Validation tests: pricing invariants and model properties"""

import math
import numpy as np
import pytest
from quantkit.pricing.core.pricer import Pricer
from quantkit.pricing.core.market import MarketData
from quantkit.pricing.core.instruments import Option, OptionType, OptionStyle

def test_american_gte_european_call(standard_market, eur_call, am_call):
    """American call option value >= European call option value"""
    eur_price = Pricer.price(eur_call, standard_market).price
    am_price = Pricer.price(am_call, standard_market).price
    
    # Due to floating point math, we allow a tiny epsilon
    assert am_price >= eur_price - 1e-5


def test_american_gte_european_put(standard_market, eur_put):
    """American put option value >= European put option value"""
    am_put = Option(strike=eur_put.strike, maturity=eur_put.maturity, 
                    option_type=OptionType.PUT, style=OptionStyle.AMERICAN)
    
    eur_price = Pricer.price(eur_put, standard_market).price
    am_price = Pricer.price(am_put, standard_market).price
    
    assert am_price >= eur_price - 1e-5


def test_put_call_parity(standard_market, eur_call, eur_put):
    """European put-call parity holds"""
    c_price = Pricer.price(eur_call, standard_market).price
    p_price = Pricer.price(eur_put, standard_market).price
    
    S = standard_market.spot
    K = eur_call.strike
    r = standard_market.rate
    T = eur_call.maturity
    
    lhs = c_price - p_price
    rhs = S - K * math.exp(-r * T)
    
    # Check that parity holds to a tight tolerance
    assert np.isclose(lhs, rhs, atol=1e-5)


# ====================================================================
# Additional Model Property Tests
# ====================================================================

def test_monotonicity_spot(standard_market, eur_call, eur_put):
    """Call prices should increase with spot; Put prices should decrease with spot."""
    market_low = MarketData(spot=90.0, rate=0.05, volatility=0.2)
    c_low = Pricer.price(eur_call, market_low).price
    p_low = Pricer.price(eur_put, market_low).price

    market_high = MarketData(spot=110.0, rate=0.05, volatility=0.2)
    c_high = Pricer.price(eur_call, market_high).price
    p_high = Pricer.price(eur_put, market_high).price

    assert c_high > c_low, "Call price did not increase with spot price"
    assert p_high < p_low, "Put price did not decrease with spot price"


def test_lower_bounds(standard_market, eur_call, eur_put):
    """European options must respect absolute theoretical lower bounds."""
    S = standard_market.spot
    K = eur_call.strike
    r = standard_market.rate
    T = eur_call.maturity

    c_price = Pricer.price(eur_call, standard_market).price
    p_price = Pricer.price(eur_put, standard_market).price

    discounted_strike = K * math.exp(-r * T)
    assert c_price >= max(0.0, S - discounted_strike) - 1e-5
    assert p_price >= max(0.0, discounted_strike - S) - 1e-5


def test_time_value_is_positive(standard_market, am_call):
    """An option's price should always be >= its immediate intrinsic value."""
    S = standard_market.spot
    K = am_call.strike
    
    price = Pricer.price(am_call, standard_market).price
    intrinsic_value = max(0.0, S - K)
    
    assert price >= intrinsic_value - 1e-5