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

# can you add tests for monotonicity with respect to strike, time to maturity, and volatility, risk free rate?


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

# upper bounds, bounds for American options, etc.


def test_time_value_is_positive(standard_market, am_call):
    """An option's price should always be >= its immediate intrinsic value."""
    S = standard_market.spot
    K = am_call.strike
    
    price = Pricer.price(am_call, standard_market).price
    intrinsic_value = max(0.0, S - K)
    
    assert price >= intrinsic_value - 1e-5

# can you add a test for deep ITM European puts? 

# ====================================================================
# Additional Monotonicity & Boundary Tests
# ====================================================================

def test_monotonicity_strike(standard_market):
    """Call prices decrease with strike; Put prices increase with strike."""
    market = standard_market
    
    call_low_k = Option(strike=90.0, maturity=1.0, option_type=OptionType.CALL, style=OptionStyle.EUROPEAN)
    call_high_k = Option(strike=110.0, maturity=1.0, option_type=OptionType.CALL, style=OptionStyle.EUROPEAN)
    
    put_low_k = Option(strike=90.0, maturity=1.0, option_type=OptionType.PUT, style=OptionStyle.EUROPEAN)
    put_high_k = Option(strike=110.0, maturity=1.0, option_type=OptionType.PUT, style=OptionStyle.EUROPEAN)

    assert Pricer.price(call_low_k, market).price > Pricer.price(call_high_k, market).price
    assert Pricer.price(put_low_k, market).price < Pricer.price(put_high_k, market).price


def test_monotonicity_volatility(standard_market, eur_call, eur_put):
    """All standard option prices should increase with volatility."""
    market_low_vol = MarketData(spot=100.0, rate=0.05, volatility=0.1)
    market_high_vol = MarketData(spot=100.0, rate=0.05, volatility=0.3)

    assert Pricer.price(eur_call, market_high_vol).price > Pricer.price(eur_call, market_low_vol).price
    assert Pricer.price(eur_put, market_high_vol).price > Pricer.price(eur_put, market_low_vol).price


def test_monotonicity_time(standard_market):
    """
    American options always gain value with more time. 
    (Note: European puts can sometimes lose value with time if deep ITM, so we test American here).
    """
    am_call_short = Option(strike=100.0, maturity=0.5, option_type=OptionType.CALL, style=OptionStyle.AMERICAN)
    am_call_long = Option(strike=100.0, maturity=1.0, option_type=OptionType.CALL, style=OptionStyle.AMERICAN)
    
    am_put_short = Option(strike=100.0, maturity=0.5, option_type=OptionType.PUT, style=OptionStyle.AMERICAN)
    am_put_long = Option(strike=100.0, maturity=1.0, option_type=OptionType.PUT, style=OptionStyle.AMERICAN)

    assert Pricer.price(am_call_long, standard_market).price >= Pricer.price(am_call_short, standard_market).price
    assert Pricer.price(am_put_long, standard_market).price >= Pricer.price(am_put_short, standard_market).price


def test_monotonicity_rate(standard_market, eur_call, eur_put):
    """Calls increase with risk-free rate; Puts decrease."""
    market_low_r = MarketData(spot=100.0, rate=0.01, volatility=0.2)
    market_high_r = MarketData(spot=100.0, rate=0.10, volatility=0.2)

    assert Pricer.price(eur_call, market_high_r).price > Pricer.price(eur_call, market_low_r).price
    assert Pricer.price(eur_put, market_high_r).price < Pricer.price(eur_put, market_low_r).price


def test_upper_bounds(standard_market, eur_call, eur_put):
    """
    Call options can never exceed the spot price.
    European Put options can never exceed the discounted strike price.
    """
    S = standard_market.spot
    K = eur_call.strike
    r = standard_market.rate
    T = eur_call.maturity

    c_price = Pricer.price(eur_call, standard_market).price
    p_price = Pricer.price(eur_put, standard_market).price

    assert c_price <= S
    assert p_price <= K * math.exp(-r * T)


def test_american_bounds(standard_market, am_call):
    """
    American options absolute upper bounds:
    American Call <= S, American Put <= K.
    """
    am_put = Option(strike=100.0, maturity=1.0, option_type=OptionType.PUT, style=OptionStyle.AMERICAN)
    
    S = standard_market.spot
    K = am_put.strike

    assert Pricer.price(am_call, standard_market).price <= S
    assert Pricer.price(am_put, standard_market).price <= K


def test_deep_itm_european_put():
    """A deep ITM European put should approximately equal K*e^(-rT) - S."""
    market = MarketData(spot=10.0, rate=0.05, volatility=0.2) # Very low spot
    K = 100.0
    T = 1.0
    r = market.rate
    
    deep_put = Option(strike=K, maturity=T, option_type=OptionType.PUT, style=OptionStyle.EUROPEAN)
    price = Pricer.price(deep_put, market).price
    
    expected_val = max(0.0, K * math.exp(-r * T) - market.spot)
    
    # It should be very close, but slightly higher due to remaining time value
    assert price >= expected_val
    assert np.isclose(price, expected_val, atol=0.1) # Tolerance for minimal time value left