import numpy as np
from quantkit.pricing.engines.binomial_tree import BinomialTreeEngine
from quantkit.pricing.engines.bsm import BSMEngine

def test_tree_converges_to_bsm(standard_market, eur_put):
    """A European Binomial Tree with many steps should match BSM."""
    bsm_price = BSMEngine.price(eur_put, standard_market).price
    
    # 500 steps should get us reasonably close to the continuous BSM
    tree = BinomialTreeEngine(steps=500, model="crr")
    tree_price = tree.price(eur_put, standard_market).price

    # Check they are within 2 cents of each other
    assert np.isclose(tree_price, bsm_price, atol=0.02)

def test_american_early_exercise_premium(standard_market, eur_put):
    """American put should be worth >= European put due to early exercise."""
    # Convert eur_put fixture to american
    from quantkit.pricing.core.instruments import OptionStyle
    am_put = eur_put
    am_put.style = OptionStyle.AMERICAN
    
    tree = BinomialTreeEngine(steps=100)
    
    am_price = tree.price(am_put, standard_market).price
    
    # Flip it back to European to get the baseline
    am_put.style = OptionStyle.EUROPEAN
    eur_price = tree.price(am_put, standard_market).price
    
    assert am_price >= eur_price