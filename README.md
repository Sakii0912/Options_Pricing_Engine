# QuantKit - Quantitative Finance Toolkit

Option pricing engine supporting multiple pricing models:
- **Black-Scholes-Merton (BSM)**: European options
- **Binomial Tree**: European and American options (CRR/JR models)
- **Least Squares Monte Carlo (LSMC)**: American and European options

## Project Structure

```
quantkit/
├── pricing/
│   ├── core/
│   │   ├── instruments.py    # Option, OptionStyle, OptionType
│   │   ├── market.py         # MarketData
│   │   └── pricer.py         # Pricing router
│   ├── engines/
│   │   ├── bsm.py            # Black-Scholes-Merton
│   │   ├── binomial_tree.py  # Binomial Tree (CRR/JR)
│   │   └── lsmc.py           # Least Squares Monte Carlo
│   ├── utils/
│   │   ├── boundary.py       # Exercise boundary utilities
│   │   └── visualization.py  # Plotting utilities
│   └── config.py             # Global hyperparameters
tests/                         # Unit tests
notebooks/                     # Jupyter notebooks
reports/                       # Documentation and analysis
```

## Installation

```bash
pip install -e .
```

## Usage

```python
from quantkit.pricing import Option, OptionStyle, OptionType, MarketData, Pricer

# Define option
option = Option(
    strike=100.0,
    maturity=1.0,
    option_type=OptionType.CALL,
    style=OptionStyle.AMERICAN
)

# Define market
market = MarketData(spot=100.0, rate=0.05, volatility=0.2)

# Price
price = Pricer.price(option, market, engine="lsmc")
```

## Development

Run tests:
```bash
pytest tests/
```

## References

- Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities.
- Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). Option pricing: A simplified approach.
- Longstaff, F. A., & Schwartz, E. S. (2001). Valuing American options via simulation.
