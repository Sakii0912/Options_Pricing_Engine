import math
import numpy as np
import pytest
from quantkit.pricing.engines.bsm import BSMEngine
from quantkit.pricing.core.instruments import OptionStyle

def test_bsm_put_call_parity(standard_market, eur_call, eur_put):
    """Test that BSM prices satisfy Put-Call parity."""
    c_res = BSMEngine.price(eur_call, standard_market)
    p_res = BSMEngine.price(eur_put, standard_market)

    C = c_res.price
    P = p_res.price
    S = standard_market.spot
    K = eur_call.strike
    r = standard_market.rate
    T = eur_call.maturity

    # Put-Call Parity: C - P = S - K * e^(-rT) (assuming q=0)
    lhs = C - P
    rhs = S - K * math.exp(-r * T)

    assert np.isclose(lhs, rhs, atol=1e-5), f"Parity failed: {lhs} != {rhs}"

def test_bsm_intrinsic_value_at_expiry(standard_market, eur_call):
    """At T=0, the option price should exactly equal its intrinsic value."""
    eur_call.maturity = 0.0  # Force expiration
    res = BSMEngine.price(eur_call, standard_market)
    
    # Spot=100, Strike=100 -> Intrinsic = 0
    assert res.price == 0.0
    
    standard_market.spot = 110.0
    res = BSMEngine.price(eur_call, standard_market)
    assert res.price == 10.0 # 110 - 100

def test_bsm_rejects_american_discrete_divs(dividend_market, am_call):
    """BSM should raise an error if asked to price American with discrete dividends."""
    with pytest.raises(ValueError, match="BSM engine does not support American options"):
        BSMEngine.price(am_call, dividend_market)