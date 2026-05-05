"""Least Squares Monte Carlo engine tests"""

import pytest
import numpy as np
from quantkit.pricing.engines.lsmc import LSMCEngine, LSMCConfig, BasisType, RegressionType
from quantkit.pricing.engines.bsm import BSMEngine
from quantkit.pricing.core.market import MarketData
from quantkit.pricing.core.instruments import Option, OptionType, OptionStyle


def test_lsmc_convergence(standard_market):
    """Test convergence as paths/steps increase"""
    eur_put = Option(strike=100.0, maturity=1.0, option_type=OptionType.PUT, style=OptionStyle.EUROPEAN)
    
    # Get exact analytical price via BSM
    exact_price = BSMEngine.price(eur_put, standard_market).price
    
    # 1. Run LSMC with very low paths/steps
    config_low = LSMCConfig(n_paths=1000, n_steps=10, seed=42)
    price_low = LSMCEngine(config_low).price(eur_put, standard_market).price
    error_low = abs(price_low - exact_price)
    
    # 2. Run LSMC with higher paths/steps
    config_high = LSMCConfig(n_paths=50000, n_steps=50, seed=42)
    price_high = LSMCEngine(config_high).price(eur_put, standard_market).price
    error_high = abs(price_high - exact_price)
    
    # The higher path run should be closer to the exact BSM price
    assert error_high < error_low


def test_lsmc_basis_functions(standard_market):
    """Test different basis function families"""
    am_put = Option(strike=100.0, maturity=1.0, option_type=OptionType.PUT, style=OptionStyle.AMERICAN)
    
    prices = {}
    for basis in BasisType:
        # Keep paths relatively low to make the test run fast
        config = LSMCConfig(n_paths=5000, n_steps=20, basis_type=basis, seed=42)
        engine = LSMCEngine(config)
        res = engine.price(am_put, standard_market)
        
        prices[basis] = res.price
        
        # Make sure it didn't crash and output a valid positive price
        assert res.price > 0, f"Failed on basis type: {basis}"
        
    # Check that Laguerre, Hermite, and Polynomial roughly agree on the price
    values = list(prices.values())
    max_diff = np.ptp(values) # Peak-to-peak (max - min)
    
    # Different polynomials shouldn't deviate by more than $1.0 on a standard $100 strike put
    assert max_diff < 1.0


# ====================================================================
# Additional LSMC Architecture Tests
# ====================================================================

@pytest.fixture
def lsmc_fast_config():
    """A lightweight LSMC config specifically for fast unit testing."""
    return LSMCConfig(n_paths=10000, n_steps=50, seed=42)


def test_lsmc_european_matches_bsm(standard_market, lsmc_fast_config):
    """LSMC European pricing should approximate exact BSM pricing."""
    eur_put = Option(strike=100.0, maturity=1.0, option_type=OptionType.PUT, style=OptionStyle.EUROPEAN)
    
    lsmc_engine = LSMCEngine(lsmc_fast_config)
    lsmc_price = lsmc_engine.price(eur_put, standard_market).price
    bsm_price = BSMEngine.price(eur_put, standard_market).price
    
    # Monte Carlo has variance; allowing a loose 25-cent tolerance
    assert np.isclose(lsmc_price, bsm_price, atol=0.25)


def test_lsmc_regression_types_execute(standard_market):
    """Ensure the LSMC engine runs successfully across all regression types (OLS, Ridge)."""
    am_put = Option(strike=100.0, maturity=1.0, option_type=OptionType.PUT, style=OptionStyle.AMERICAN)
    
    for reg in RegressionType:
        config = LSMCConfig(n_paths=2000, n_steps=10, regression_type=reg, seed=42)
        engine = LSMCEngine(config)
        res = engine.price(am_put, standard_market)
        
        assert res.price > 0, f"Failed on regression type: {reg}"

# LSMC - European put <= American Put
# LSMC - American call = European call (no dividends) (with tolerance)
# LSMC - put call parity with tolerance 
# LSMC - test bounds for EACH with some tolerance
# monotonocity tests 

# ====================================================================
# LSMC Specific Invariant & Property Tests
# ====================================================================

def test_lsmc_american_call_equals_european(standard_market, lsmc_fast_config):
    """For a non-dividend paying stock, American Call == European Call."""
    # LSMC has variance, so we use the exact same seed and config for both to ensure
    # the paths generated are 100% identical.
    am_call = Option(strike=100.0, maturity=1.0, option_type=OptionType.CALL, style=OptionStyle.AMERICAN)
    eur_call = Option(strike=100.0, maturity=1.0, option_type=OptionType.CALL, style=OptionStyle.EUROPEAN)
    
    lsmc_engine = LSMCEngine(lsmc_fast_config)
    
    am_price = lsmc_engine.price(am_call, standard_market).price
    eur_price = lsmc_engine.price(eur_call, standard_market).price
    
    # Because there are no dividends, early exercise is never optimal.
    # The regression should figure this out and yield essentially the same price.
    assert np.isclose(am_price, eur_price, atol=0.05)


def test_lsmc_put_call_parity(standard_market, lsmc_fast_config):
    """European put-call parity holds for LSMC within Monte Carlo tolerance."""
    eur_call = Option(strike=100.0, maturity=1.0, option_type=OptionType.CALL, style=OptionStyle.EUROPEAN)
    eur_put = Option(strike=100.0, maturity=1.0, option_type=OptionType.PUT, style=OptionStyle.EUROPEAN)
    
    lsmc_engine = LSMCEngine(lsmc_fast_config)
    c_price = lsmc_engine.price(eur_call, standard_market).price
    p_price = lsmc_engine.price(eur_put, standard_market).price
    
    S = standard_market.spot
    K = eur_call.strike
    r = standard_market.rate
    T = eur_call.maturity
    
    lhs = c_price - p_price
    rhs = S - K * np.exp(-r * T)
    
    # Monte Carlo variance means parity won't be exact to the 5th decimal.
    # 50 cents of tolerance is reasonable for 10k paths.
    assert np.isclose(lhs, rhs, atol=0.5)


def test_lsmc_bounds(standard_market, lsmc_fast_config):
    """LSMC prices must respect absolute lower and upper bounds."""
    am_put = Option(strike=100.0, maturity=1.0, option_type=OptionType.PUT, style=OptionStyle.AMERICAN)
    am_call = Option(strike=100.0, maturity=1.0, option_type=OptionType.CALL, style=OptionStyle.AMERICAN)
    
    lsmc_engine = LSMCEngine(lsmc_fast_config)
    p_price = lsmc_engine.price(am_put, standard_market).price
    c_price = lsmc_engine.price(am_call, standard_market).price
    
    S = standard_market.spot
    K = am_put.strike
    r = standard_market.rate
    T = am_put.maturity
    
    # Lower bounds (Intrinsic value)
    assert p_price >= max(0.0, K - S) - 1e-4
    assert c_price >= max(0.0, S - K) - 1e-4
    
    # Upper bounds
    assert p_price <= K
    assert c_price <= S


def test_lsmc_monotonicity(standard_market):
    """Test LSMC monotonicity for Spot and Volatility using a locked seed."""
    am_put = Option(strike=100.0, maturity=1.0, option_type=OptionType.PUT, style=OptionStyle.AMERICAN)
    
    # Use a solid number of paths to ensure the macro-trend overwhelms any MC noise
    base_config = LSMCConfig(n_paths=20000, n_steps=50, seed=123)
    
    # 1. Spot Monotonicity (Put should decrease as Spot increases)
    market_low_s = MarketData(spot=90.0, rate=0.05, volatility=0.2)
    market_high_s = MarketData(spot=110.0, rate=0.05, volatility=0.2)
    
    p_low_s = LSMCEngine(base_config).price(am_put, market_low_s).price
    p_high_s = LSMCEngine(base_config).price(am_put, market_high_s).price
    assert p_low_s > p_high_s
    
    # 2. Volatility Monotonicity (Put should increase as Vol increases)
    market_low_v = MarketData(spot=100.0, rate=0.05, volatility=0.1)
    market_high_v = MarketData(spot=100.0, rate=0.05, volatility=0.4)
    
    p_low_v = LSMCEngine(base_config).price(am_put, market_low_v).price
    p_high_v = LSMCEngine(base_config).price(am_put, market_high_v).price
    assert p_high_v > p_low_v

def test_lsmc_european_put_leq_american(standard_market, lsmc_fast_config):
    """LSMC: American put price should be >= European put price."""
    # Define both options with identical parameters except for style
    am_put = Option(strike=100.0, maturity=1.0, option_type=OptionType.PUT, style=OptionStyle.AMERICAN)
    eur_put = Option(strike=100.0, maturity=1.0, option_type=OptionType.PUT, style=OptionStyle.EUROPEAN)

    lsmc_engine = LSMCEngine(lsmc_fast_config)
    
    # Price both using the exact same configuration and seed
    am_price = lsmc_engine.price(am_put, standard_market).price
    eur_price = lsmc_engine.price(eur_put, standard_market).price

    # American should always be greater than or equal to European.
    # We include a tiny 1e-4 buffer just in case float arithmetic gets weird.
    assert am_price >= eur_price - 1e-4