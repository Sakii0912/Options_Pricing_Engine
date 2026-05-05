import copy
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from quantkit.pricing.core.instruments import Option, OptionType, OptionStyle
from quantkit.pricing.core.market import MarketData
from quantkit.pricing.core.pricer import Pricer

import time
from quantkit.pricing.core.market import DividendEvent

# ==========================================
# 5. Convergence Error & Runtime Analysis
# ==========================================
def plot_convergence_error_and_time(option: Option, market: MarketData, step_range: range):
    """
    Plots the absolute pricing error (Binomial vs BSM) against Steps, 
    and the Runtime vs Pricing Error.
    """
    bsm_price = Pricer.price(option, market, engine="bsm").price
    
    steps_list = list(step_range)
    errors = []
    runtimes = []
    
    for steps in steps_list:
        start_time = time.time()
        tree_price = Pricer.price(option, market, engine="binomial", steps=steps).price
        end_time = time.time()
        
        errors.append(abs(tree_price - bsm_price))
        runtimes.append(end_time - start_time)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Steps vs Error
    ax1.plot(steps_list, errors, marker='o', color='red', markersize=4)
    ax1.set_title('Binomial Tree: Steps vs Absolute Error (vs BSM)')
    ax1.set_xlabel('Number of Steps (N)')
    ax1.set_ylabel('Absolute Error')
    ax1.grid(True, linestyle='--')
    
    # Plot 2: Runtime vs Error
    ax2.scatter(runtimes, errors, color='purple')
    ax2.set_title('Computational Complexity: Runtime vs Error')
    ax2.set_xlabel('Runtime (seconds)')
    ax2.set_ylabel('Absolute Error')
    ax2.grid(True, linestyle='--')
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 6. Put-Call Parity Table
# ==========================================
def generate_put_call_parity_table(market: MarketData, examples: list):
    """
    Evaluates European Put-Call Parity: C - P = S - PV(K) - PV(Divs)
    """
    results = []
    for opt_params in examples:
        K = opt_params['strike']
        T = opt_params['maturity']
        
        call = Option(strike=K, maturity=T, option_type=OptionType.CALL, style=OptionStyle.EUROPEAN)
        put = Option(strike=K, maturity=T, option_type=OptionType.PUT, style=OptionStyle.EUROPEAN)
        
        c_price = Pricer.price(call, market, engine="bsm").price
        p_price = Pricer.price(put, market, engine="bsm").price
        
        # Calculate Right Hand Side
        pv_k = K * math.exp(-market.rate * T)
        pv_div = market.pv_discrete_dividends(T) if hasattr(market, 'pv_discrete_dividends') else 0.0
        
        lhs = c_price - p_price
        rhs = market.spot - pv_k - pv_div
        
        results.append({
            "Strike": K,
            "Maturity": T,
            "Call Price": round(c_price, 4),
            "Put Price": round(p_price, 4),
            "C - P (LHS)": round(lhs, 4),
            "S - PV(K) - PV(Div) (RHS)": round(rhs, 4),
            "Diff (Error)": round(abs(lhs - rhs), 6)
        })
        
    return pd.DataFrame(results)

# ==========================================
# 7. Price Bounds Plotting
# ==========================================
def plot_price_bounds(base_option: Option, base_market: MarketData, spot_range: np.ndarray, engine: str="auto", **kwargs):
    """
    Plots the Option Price alongside its strict theoretical upper and lower bounds as Spot varies.
    Adjusted for continuous yields and discrete dividends.
    """
    prices, lower_bounds, upper_bounds = [], [], []
    r = base_market.rate
    q = base_market.dividend_yield
    T = base_option.maturity
    K = base_option.strike
    
    for S in spot_range:
        market = copy.deepcopy(base_market)
        market.spot = S
        
        price = Pricer.price(base_option, market, engine=engine, **kwargs).price
        prices.append(price)
        
        # --- BOUNDS MATH CORRECTION ---
        # 1. Adjust Spot for continuous yield
        adj_S = S * math.exp(-q * T)
        
        # 2. Adjust Spot for discrete dividends (if applicable)
        if hasattr(market, 'has_discrete_dividends') and market.has_discrete_dividends(T):
            adj_S -= market.pv_discrete_dividends(T)
            
        pv_k = K * math.exp(-r * T)
        
        if base_option.option_type == OptionType.CALL:
            # Lower bound: max(0, S*e^-qT - PV(Divs) - PV(K))
            lower_bounds.append(max(0.0, adj_S - pv_k))
            
            # Upper bound: S*e^-qT - PV(Divs)
            upper_bounds.append(adj_S)
        else: # PUT
            # Lower bound: max(0, PV(K) - S*e^-qT + PV(Divs))
            lower_bounds.append(max(0.0, pv_k - adj_S))
            
            # Upper bound: K (American) or PV(K) (European)
            upper_bounds.append(K if base_option.style == OptionStyle.AMERICAN else pv_k)
            
    plt.figure(figsize=(10, 6))
    plt.plot(spot_range, upper_bounds, 'k--', label='Upper Bound')
    plt.plot(spot_range, lower_bounds, 'r--', label='Lower Bound')
    plt.plot(spot_range, prices, 'b-', linewidth=2, label=f'Price ({engine})')
    
    # Fill out-of-bounds areas with red
    plt.fill_between(spot_range, upper_bounds, max(upper_bounds)*1.5, color='red', alpha=0.1)
    plt.fill_between(spot_range, 0, lower_bounds, color='red', alpha=0.1)
    
    plt.title(f'Theoretical Bounds: {base_option.style.value.capitalize()} {base_option.option_type.value.capitalize()}')
    plt.xlabel('Spot Price (S)')
    plt.ylabel('Option Price')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.show()

# ==========================================
# 8. Time-Series Path Simulation
# ==========================================
def plot_price_over_time_path(base_option: Option, base_market: MarketData, days: int = 20, sigmas: list = [0.2]):
    """
    Simulates a stock path over 'days' and plots the resulting option price over time.
    """
    dt = 1.0 / 252.0
    np.random.seed(42) # For reproducible paths
    
    plt.figure(figsize=(12, 6))
    
    for sigma in sigmas:
        market = copy.deepcopy(base_market)
        option = copy.deepcopy(base_option)
        market.volatility = sigma
        
        prices = []
        spots = [market.spot]
        
        # Generate Random Walk
        for _ in range(days):
            dW = np.random.normal(0, math.sqrt(dt))
            drift = (market.rate - market.dividend_yield - 0.5 * sigma**2) * dt
            shock = sigma * dW
            spots.append(spots[-1] * math.exp(drift + shock))
            
        # Calculate Option Price at each day
        for i, S in enumerate(spots):
            market.spot = S
            option.maturity = max(1e-5, base_option.maturity - (i * dt))
            prices.append(Pricer.price(option, market, engine="bsm").price)
            
        plt.plot(range(days + 1), prices, label=f'Option Price (Vol={sigma*100}%)')
        
    plt.title('Option Price Evolution over 20 Days Simulation')
    plt.xlabel('Days Passed')
    plt.ylabel('Option Price')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.show()

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