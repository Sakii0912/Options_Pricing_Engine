import copy
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from quantkit.pricing.core.instruments import Option, OptionType, OptionStyle
from quantkit.pricing.core.market import MarketData
from quantkit.pricing.core.pricer import Pricer

# ==========================================
# 1. Sensitivities (Greeks Proxy)
# ==========================================
def plot_sensitivities(base_option: Option, base_market: MarketData, param_name: str, param_range: np.ndarray, engine: str = "auto", **kwargs):
    """
    Plots the variation of option price with respect to a given parameter.
    param_name must be one of: 'spot', 'volatility', 'rate', 'maturity'
    """
    prices = []
    
    for val in param_range:
        # Clone to avoid mutating the original objects
        market = copy.deepcopy(base_market)
        option = copy.deepcopy(base_option)
        
        if param_name == 'spot':
            market.spot = val
        elif param_name == 'volatility':
            market.volatility = val
        elif param_name == 'rate':
            market.rate = val
        elif param_name == 'maturity':
            option.maturity = val
        else:
            raise ValueError(f"Unsupported parameter: {param_name}")
            
        res = Pricer.price(option, market, engine=engine, **kwargs)
        prices.append(res.price)
        
    plt.figure(figsize=(8, 5))
    plt.plot(param_range, prices, label=f'{option.style.value.capitalize()} {option.option_type.value.capitalize()}', color='blue', linewidth=2)
    plt.title(f'Option Price vs {param_name.capitalize()}')
    plt.xlabel(param_name.capitalize())
    plt.ylabel('Price')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

# ==========================================
# 2. Option Price Bounds Table
# ==========================================
def generate_bounds_table(options_list: list, market: MarketData, engine: str = "auto"):
    """
    Generates a DataFrame showing that option prices fall strictly within theoretical bounds.
    """
    results = []
    S = market.spot
    r = market.rate
    
    for opt in options_list:
        K = opt.strike
        T = opt.maturity
        
        # Calculate theoretical bounds
        discounted_K = K * math.exp(-r * T)
        
        if opt.option_type == OptionType.CALL:
            lower_bound = max(0.0, S - discounted_K)
            upper_bound = S
        else: # PUT
            lower_bound = max(0.0, discounted_K - S)
            upper_bound = K if opt.style == OptionStyle.AMERICAN else discounted_K
            
        price = Pricer.price(opt, market, engine=engine).price
        
        results.append({
            "Style": opt.style.value.capitalize(),
            "Type": opt.option_type.value.capitalize(),
            "Strike": K,
            "Maturity": T,
            "Lower Bound": round(lower_bound, 4),
            "Actual Price": round(price, 4),
            "Upper Bound": round(upper_bound, 4),
            "Valid?": lower_bound - 1e-5 <= price <= upper_bound + 1e-5 # accounting for float precision
        })
        
    df = pd.DataFrame(results)
    return df

# ==========================================
# 3. American vs European Premium Table
# ==========================================
def generate_american_vs_european_table(market: MarketData, strikes: list, maturities: list, opt_type: OptionType):
    """
    Generates a table comparing American vs European prices to prove American >= European.
    """
    results = []
    
    for K in strikes:
        for T in maturities:
            am_opt = Option(strike=K, maturity=T, option_type=opt_type, style=OptionStyle.AMERICAN)
            eu_opt = Option(strike=K, maturity=T, option_type=opt_type, style=OptionStyle.EUROPEAN)
            
            # Using binomial tree to keep comparison apples-to-apples if desired, or auto
            am_price = Pricer.price(am_opt, market, engine="binomial", steps=150).price
            eu_price = Pricer.price(eu_opt, market, engine="binomial", steps=150).price
            
            results.append({
                "Type": opt_type.value.capitalize(),
                "Strike": K,
                "Maturity": T,
                "European Price": round(eu_price, 4),
                "American Price": round(am_price, 4),
                "Early Exercise Premium": round(am_price - eu_price, 4),
                "Invariant Holds?": am_price >= eu_price - 1e-5
            })
            
    return pd.DataFrame(results)

# ==========================================
# 4. Binomial Tree Convergence Graph
# ==========================================
def plot_binomial_convergence(option: Option, market: MarketData, step_range: range):
    """
    Plots the convergence of the Binomial Tree to the BSM price as steps increase.
    Requires a European option for direct comparison to closed-form BSM.
    """
    if option.style != OptionStyle.EUROPEAN:
        raise ValueError("Convergence tests to BSM require a European Option.")
        
    bsm_price = Pricer.price(option, market, engine="bsm").price
    tree_prices = []
    
    # Extract steps to a list for plotting
    steps_list = list(step_range)
    
    for steps in steps_list:
        price = Pricer.price(option, market, engine="binomial", steps=steps).price
        tree_prices.append(price)
        
    plt.figure(figsize=(10, 6))
    plt.plot(steps_list, tree_prices, label='Binomial Tree Price', marker='o', markersize=3, alpha=0.7)
    plt.axhline(y=bsm_price, color='red', linestyle='-', label=f'BSM Price ({bsm_price:.4f})')
    
    plt.title('Binomial Tree Convergence to BSM')
    plt.xlabel('Number of Steps (N)')
    plt.ylabel('Option Price')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()