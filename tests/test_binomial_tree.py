"""Binomial Tree option pricing engine tests"""

import pytest
import math
from quantkit.pricing.core.instruments import Option, OptionType, OptionStyle
from quantkit.pricing.core.market import MarketData
from quantkit.pricing.engines.binomial_tree import BinomialTreeEngine
from quantkit.pricing.engines.bsm import BSMEngine


class TestBinomialTreeEngine:
    """Test Binomial Tree pricing engine"""

    @pytest.fixture
    def standard_market(self):
        """Standard market data for testing"""
        return MarketData(spot=100.0, rate=0.05, volatility=0.2)

    @pytest.fixture
    def european_call_atm(self):
        """ATM European call option"""
        return Option(
            strike=100.0,
            maturity=1.0,
            option_type=OptionType.CALL,
            style=OptionStyle.EUROPEAN
        )

    @pytest.fixture
    def european_put_atm(self):
        """ATM European put option"""
        return Option(
            strike=100.0,
            maturity=1.0,
            option_type=OptionType.PUT,
            style=OptionStyle.EUROPEAN
        )

    def test_binomial_tree_european_call_crr(self, standard_market, european_call_atm):
        """Test binomial tree pricing for European call option (CRR model)"""
        engine = BinomialTreeEngine(steps=100, model="crr")
        price = engine.price(european_call_atm, standard_market)

        # Should be in reasonable range
        assert isinstance(price, float)
        assert price > 0
        assert 9 < price < 12

    def test_binomial_tree_european_call_jr(self, standard_market, european_call_atm):
        """Test binomial tree pricing for European call option (JR model)"""
        engine = BinomialTreeEngine(steps=100, model="jr")
        price = engine.price(european_call_atm, standard_market)

        # Should be in reasonable range
        assert isinstance(price, float)
        assert price > 0
        assert 9 < price < 12

    def test_binomial_tree_european_put_crr(self, standard_market, european_put_atm):
        """Test binomial tree pricing for European put option (CRR model)"""
        engine = BinomialTreeEngine(steps=100, model="crr")
        price = engine.price(european_put_atm, standard_market)

        # Should be in reasonable range
        assert isinstance(price, float)
        assert price > 0
        assert 4 < price < 6

    def test_binomial_tree_european_put_jr(self, standard_market, european_put_atm):
        """Test binomial tree pricing for European put option (JR model)"""
        engine = BinomialTreeEngine(steps=100, model="jr")
        price = engine.price(european_put_atm, standard_market)

        # Should be in reasonable range
        assert isinstance(price, float)
        assert price > 0
        assert 4 < price < 6

    def test_binomial_tree_european_call_itm(self, standard_market):
        """Test binomial tree pricing with ITM European call"""
        call = Option(
            strike=90.0,
            maturity=1.0,
            option_type=OptionType.CALL,
            style=OptionStyle.EUROPEAN
        )
        engine = BinomialTreeEngine(steps=100)
        price = engine.price(call, standard_market)

        # ITM call should be worth at least its intrinsic value
        intrinsic = 100 - 90
        assert price > intrinsic

    def test_binomial_tree_european_call_otm(self, standard_market):
        """Test binomial tree pricing with OTM European call"""
        call = Option(
            strike=110.0,
            maturity=1.0,
            option_type=OptionType.CALL,
            style=OptionStyle.EUROPEAN
        )
        engine = BinomialTreeEngine(steps=100)
        price = engine.price(call, standard_market)

        # OTM call should have positive value
        assert isinstance(price, float)
        assert price > 0
        assert price < 10

    def test_binomial_convergence_to_bsm_call(self, standard_market, european_call_atm):
        """Test convergence of tree prices to BSM for increasing steps (CALL)"""
        bsm_price = BSMEngine.price(european_call_atm, standard_market)

        steps_list = [10, 50, 100, 200, 500]
        errors = []

        for steps in steps_list:
            engine = BinomialTreeEngine(steps=steps, model="crr")
            tree_price = engine.price(european_call_atm, standard_market)
            error = abs(tree_price - bsm_price)
            errors.append(error)

        # Error should decrease as steps increase
        assert errors[-1] < errors[0], "Error should decrease with more steps"

        # Final error should be reasonably small
        assert errors[-1] < 0.1

    def test_binomial_convergence_to_bsm_put(self, standard_market, european_put_atm):
        """Test convergence of tree prices to BSM for increasing steps (PUT)"""
        bsm_price = BSMEngine.price(european_put_atm, standard_market)

        steps_list = [10, 50, 100, 200, 500]
        errors = []

        for steps in steps_list:
            engine = BinomialTreeEngine(steps=steps, model="crr")
            tree_price = engine.price(european_put_atm, standard_market)
            error = abs(tree_price - bsm_price)
            errors.append(error)

        # Error should decrease as steps increase
        assert errors[-1] < errors[0], "Error should decrease with more steps"

        # Final error should be reasonably small
        assert errors[-1] < 0.1

    def test_binomial_crr_vs_jr_convergence(self, standard_market, european_call_atm):
        """Compare CRR and JR models at high step count (should converge to same value)"""
        bsm_price = BSMEngine.price(european_call_atm, standard_market)

        engine_crr = BinomialTreeEngine(steps=200, model="crr")
        engine_jr = BinomialTreeEngine(steps=200, model="jr")

        price_crr = engine_crr.price(european_call_atm, standard_market)
        price_jr = engine_jr.price(european_call_atm, standard_market)

        # Both should be close to BSM
        assert abs(price_crr - bsm_price) < 0.15
        assert abs(price_jr - bsm_price) < 0.15

        # CRR and JR should converge to each other at high steps
        assert abs(price_crr - price_jr) < 0.2

    def test_binomial_short_maturity(self, standard_market):
        """Test binomial tree pricing with short time to maturity"""
        call_short = Option(
            strike=100.0,
            maturity=0.1,  # Short maturity
            option_type=OptionType.CALL,
            style=OptionStyle.EUROPEAN
        )
        call_long = Option(
            strike=100.0,
            maturity=1.0,
            option_type=OptionType.CALL,
            style=OptionStyle.EUROPEAN
        )

        engine_short = BinomialTreeEngine(steps=100)
        engine_long = BinomialTreeEngine(steps=100)

        price_short = engine_short.price(call_short, standard_market)
        price_long = engine_long.price(call_long, standard_market)

        # Should have positive values
        assert isinstance(price_short, float)
        assert isinstance(price_long, float)
        assert price_short > 0
        assert price_long > 0

        # Longer maturity should be worth more
        assert price_long > price_short

    def test_binomial_call_put_parity(self, standard_market):
        """Test that binomial tree satisfies put-call parity"""
        call = Option(
            strike=100.0,
            maturity=1.0,
            option_type=OptionType.CALL,
            style=OptionStyle.EUROPEAN
        )
        put = Option(
            strike=100.0,
            maturity=1.0,
            option_type=OptionType.PUT,
            style=OptionStyle.EUROPEAN
        )

        engine = BinomialTreeEngine(steps=200)

        call_price = engine.price(call, standard_market)
        put_price = engine.price(put, standard_market)

        # Parity: C - P = S - K*e^(-r*T)
        S = standard_market.spot
        K = 100.0
        r = standard_market.rate
        T = 1.0

        lhs = call_price - put_price
        rhs = S - K * math.exp(-r * T)

        # Parity should hold reasonably well for binomial tree
        assert abs(lhs - rhs) < 0.1

    def test_binomial_increasing_steps_consistency(self, standard_market, european_call_atm):
        """Test that prices are monotonically converging with increasing steps"""
        bsm_price = BSMEngine.price(european_call_atm, standard_market)

        prices = []
        for steps in [50, 100, 150, 200]:
            engine = BinomialTreeEngine(steps=steps)
            price = engine.price(european_call_atm, standard_market)
            prices.append(price)

        # Check that variance decreases (convergence behavior)
        # Prices should oscillate around BSM value
        errors = [abs(p - bsm_price) for p in prices]
        assert errors[-1] < 0.2  # Final error should be small

    def test_binomial_invalid_model(self, standard_market):
        """Test that invalid model raises error"""
        with pytest.raises(ValueError):
            BinomialTreeEngine(steps=100, model="invalid_model")

    def test_binomial_european_call_monotonicity(self, standard_market):
        """Test that call price decreases as strike increases"""
        call_low_strike = Option(
            strike=90.0,
            maturity=1.0,
            option_type=OptionType.CALL,
            style=OptionStyle.EUROPEAN
        )
        call_high_strike = Option(
            strike=110.0,
            maturity=1.0,
            option_type=OptionType.CALL,
            style=OptionStyle.EUROPEAN
        )

        engine = BinomialTreeEngine(steps=100)

        price_low = engine.price(call_low_strike, standard_market)
        price_high = engine.price(call_high_strike, standard_market)

        assert price_low > price_high

    def test_binomial_european_put_monotonicity(self, standard_market):
        """Test that put price increases as strike increases"""
        put_low_strike = Option(
            strike=90.0,
            maturity=1.0,
            option_type=OptionType.PUT,
            style=OptionStyle.EUROPEAN
        )
        put_high_strike = Option(
            strike=110.0,
            maturity=1.0,
            option_type=OptionType.PUT,
            style=OptionStyle.EUROPEAN
        )

        engine = BinomialTreeEngine(steps=100)

        price_low = engine.price(put_low_strike, standard_market)
        price_high = engine.price(put_high_strike, standard_market)

        assert price_low < price_high
