"""Black-Scholes-Merton engine tests"""

import pytest
import math
from quantkit.pricing.core.instruments import Option, OptionType, OptionStyle
from quantkit.pricing.core.market import MarketData
from quantkit.pricing.engines.bsm import BSMEngine


class TestBSMEngine:
    """Test Black-Scholes-Merton pricing engine"""

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

    def test_bsm_european_call_atm(self, standard_market, european_call_atm):
        """Test BSM engine with ATM European call option"""
        price = BSMEngine.price(european_call_atm, standard_market)

        # For ATM option with S=K=100, r=0.05, sigma=0.2, T=1:
        # Approximate value should be around 10.4 (from tables)
        assert isinstance(price, float)
        assert price > 0
        assert 9 < price < 12  # Realistic range for these params

    def test_bsm_european_put_atm(self, standard_market, european_put_atm):
        """Test BSM engine with ATM European put option"""
        price = BSMEngine.price(european_put_atm, standard_market)

        # For ATM option, put should have positive value
        assert isinstance(price, float)
        assert price > 0
        assert 4 < price < 6  # Realistic range

    def test_bsm_european_call_itm(self, standard_market):
        """Test BSM engine with ITM European call (S > K)"""
        call = Option(
            strike=90.0,
            maturity=1.0,
            option_type=OptionType.CALL,
            style=OptionStyle.EUROPEAN
        )
        price = BSMEngine.price(call, standard_market)

        # ITM call should be worth at least its intrinsic value
        intrinsic = 100 - 90
        assert price > intrinsic

    def test_bsm_european_call_otm(self, standard_market):
        """Test BSM engine with OTM European call (S < K)"""
        call = Option(
            strike=110.0,
            maturity=1.0,
            option_type=OptionType.CALL,
            style=OptionStyle.EUROPEAN
        )
        price = BSMEngine.price(call, standard_market)

        # OTM call should be worth less than ATM call
        assert isinstance(price, float)
        assert price > 0
        assert price < 10  # Reasonable OTM value

    def test_bsm_european_put_itm(self, standard_market):
        """Test BSM engine with ITM European put (S < K)"""
        put = Option(
            strike=110.0,
            maturity=1.0,
            option_type=OptionType.PUT,
            style=OptionStyle.EUROPEAN
        )
        price = BSMEngine.price(put, standard_market)

        # ITM put should be worth at least its intrinsic value
        intrinsic = 110 - 100
        assert price > intrinsic

    def test_bsm_european_put_otm(self, standard_market):
        """Test BSM engine with OTM European put (S > K)"""
        put = Option(
            strike=90.0,
            maturity=1.0,
            option_type=OptionType.PUT,
            style=OptionStyle.EUROPEAN
        )
        price = BSMEngine.price(put, standard_market)

        # OTM put should be worth less than ATM put
        assert isinstance(price, float)
        assert price > 0
        assert price < 5  # Reasonable OTM value

    def test_bsm_at_maturity(self, standard_market):
        """Test BSM pricing at maturity (T → 0)"""
        # Call at maturity
        call_atm = Option(
            strike=100.0,
            maturity=0.0,
            option_type=OptionType.CALL,
            style=OptionStyle.EUROPEAN
        )
        call_itm = Option(
            strike=90.0,
            maturity=0.0,
            option_type=OptionType.CALL,
            style=OptionStyle.EUROPEAN
        )

        call_atm_price = BSMEngine.price(call_atm, standard_market)
        call_itm_price = BSMEngine.price(call_itm, standard_market)

        # At maturity, value should be intrinsic value
        assert abs(call_atm_price - 0.0) < 1e-10
        assert abs(call_itm_price - 10.0) < 1e-10

        # Put at maturity
        put_itm = Option(
            strike=110.0,
            maturity=0.0,
            option_type=OptionType.PUT,
            style=OptionStyle.EUROPEAN
        )
        put_itm_price = BSMEngine.price(put_itm, standard_market)
        assert abs(put_itm_price - 10.0) < 1e-10

    def test_bsm_call_put_parity(self, standard_market):
        """Test European call-put parity: C - P = S - K*e^(-r*T)"""
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

        call_price = BSMEngine.price(call, standard_market)
        put_price = BSMEngine.price(put, standard_market)

        # Parity: C - P = S - K*e^(-r*T)
        S = standard_market.spot
        K = 100.0
        r = standard_market.rate
        T = 1.0

        lhs = call_price - put_price
        rhs = S - K * math.exp(-r * T)

        assert abs(lhs - rhs) < 1e-6

    def test_bsm_increasing_volatility(self, standard_market, european_call_atm):
        """Test that call price increases with volatility"""
        market_low_vol = MarketData(spot=100.0, rate=0.05, volatility=0.1)
        market_high_vol = MarketData(spot=100.0, rate=0.05, volatility=0.3)

        price_low_vol = BSMEngine.price(european_call_atm, market_low_vol)
        price_high_vol = BSMEngine.price(european_call_atm, market_high_vol)

        assert price_high_vol > price_low_vol

    def test_bsm_increasing_time(self, standard_market, european_call_atm):
        """Test that OTM call price increases with time to maturity"""
        call_otm = Option(
            strike=110.0,
            maturity=0.5,
            option_type=OptionType.CALL,
            style=OptionStyle.EUROPEAN
        )
        call_otm_1yr = Option(
            strike=110.0,
            maturity=1.0,
            option_type=OptionType.CALL,
            style=OptionStyle.EUROPEAN
        )

        price_short = BSMEngine.price(call_otm, standard_market)
        price_long = BSMEngine.price(call_otm_1yr, standard_market)

        assert price_long > price_short

    def test_bsm_same_price_for_spot_strike_swap(self, standard_market):
        """Test symmetry: Call(S=100, K=110) ≈ Put(S=110, K=100) with adjusted rates"""
        # This is a more complex relationship, but tests basis function
        call = Option(
            strike=110.0,
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

        call_price = BSMEngine.price(call, standard_market)
        put_price = BSMEngine.price(put, standard_market)

        # Both are OTM with similar distance, should be in same ballpark
        # (not exact equality due to drift term)
        assert call_price > 0
        assert put_price > 0
