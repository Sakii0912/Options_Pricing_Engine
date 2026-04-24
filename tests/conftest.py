import pytest
from quantkit.pricing.core.instruments import Option, OptionType, OptionStyle
from quantkit.pricing.core.market import MarketData, DividendEvent

@pytest.fixture
def standard_market():
    """A standard Black-Scholes market environment."""
    return MarketData(spot=100.0, rate=0.05, volatility=0.2, dividend_yield=0.0)

@pytest.fixture
def dividend_market():
    """Market with a discrete dividend."""
    divs = [DividendEvent(time=0.5, amount=5.0)]
    return MarketData(spot=100.0, rate=0.05, volatility=0.2, discrete_dividends=divs)

@pytest.fixture
def eur_call():
    return Option(strike=100.0, maturity=1.0, option_type=OptionType.CALL, style=OptionStyle.EUROPEAN)

@pytest.fixture
def eur_put():
    return Option(strike=100.0, maturity=1.0, option_type=OptionType.PUT, style=OptionStyle.EUROPEAN)

@pytest.fixture
def am_call():
    return Option(strike=100.0, maturity=1.0, option_type=OptionType.CALL, style=OptionStyle.AMERICAN)