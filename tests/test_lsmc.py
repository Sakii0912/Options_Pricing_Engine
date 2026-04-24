"""Least Squares Monte Carlo engine tests"""

import pytest
import numpy as np
from quantkit.pricing.engines.lsmc import LSMCEngine, LSMCConfig, BasisType, RegressionType
from quantkit.pricing.engines.bsm import BSMEngine
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