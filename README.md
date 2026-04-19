# QuantKit - Option Pricing Engine

QuantKit is a Python toolkit for pricing vanilla options with multiple numerical methods.

## Implemented Engines

- Black-Scholes-Merton (BSM)
    - European call/put pricing
    - Continuous dividend yield support
    - Spot-adjustment approximation for European options with discrete dividends
- Binomial Tree
    - European and American call/put pricing
    - CRR and JR tree models
    - Continuous dividend yield and discrete dividend event handling
- Least Squares Monte Carlo (LSMC)
    - European and American pricing
    - Configurable basis: polynomial, Laguerre, Hermite
    - Configurable regression: OLS or Ridge
    - Returns price + estimated exercise boundary

## Current Routing Behavior

`Pricer.price(...)` currently supports `engine="auto" | "bsm" | "binomial"`.

- `auto` chooses BSM for most European no-discrete-dividend cases, otherwise Binomial
- LSMC is available as a separate engine class (`LSMCEngine`) and is not wired into `Pricer`

## Package Layout

```
quantkit/
    pricing/
        core/            # Option/Market dataclasses + routing interface
        engines/         # bsm.py, binomial_tree.py, lsmc.py
        utils/           # visualization + boundary helpers
        config.py        # global numeric defaults
tests/               # test modules (partially implemented)
notebooks/           # demos/analysis
reports/             # report drafts
```

## Installation

```bash
pip install -e .
```

Python requirement: `>=3.8`

Core dependencies:
- numpy
- scipy
- matplotlib
- pandas

Dev extras:
- pytest
- jupyter

## Quick Start

### 1) Route via `Pricer` (BSM/Binomial)

```python
from quantkit.pricing import Option, OptionStyle, OptionType, MarketData, Pricer

option = Option(
        strike=100.0,
        maturity=1.0,
        option_type=OptionType.PUT,
        style=OptionStyle.AMERICAN,
)

market = MarketData(
        spot=100.0,
        rate=0.05,
        volatility=0.2,
        dividend_yield=0.0,
)

price = Pricer.price(option, market, engine="auto", steps=200, model="crr")
print(price)
```

### 2) Use LSMC directly

```python
from quantkit.pricing import (
        Option,
        OptionStyle,
        OptionType,
        MarketData,
        LSMCEngine,
        LSMCConfig,
        BasisType,
        RegressionType,
)

option = Option(100.0, 1.0, OptionType.PUT, OptionStyle.AMERICAN)
market = MarketData(spot=100.0, rate=0.05, volatility=0.2)

config = LSMCConfig(
        n_paths=50000,
        n_steps=100,
        basis_type=BasisType.LAGUERRE,
        degree=3,
        regression_type=RegressionType.OLS,
)

engine = LSMCEngine(config)
result = engine.price(option, market)

print(result.price)
```

## Testing

Run:

```bash
pytest tests/
```

Note: several test modules are currently placeholders (`NotImplementedError`) and need completion.

## Project Status Notes

- `plot_exercise_boundaries` is implemented in visualization utilities
- Some utility/test/report files are scaffolds marked TODO/NotImplemented

## References

- Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities.
- Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). Option pricing: A simplified approach.
- Longstaff, F. A., & Schwartz, E. S. (2001). Valuing American options via simulation.
