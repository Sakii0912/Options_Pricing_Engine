"""Least Squares Monte Carlo option pricing engine"""

from ..core.instruments import *
from ..core.market import *
from dataclasses import dataclass
from numpy.polynomial.laguerre import lagvander 
from numpy.polynomial.polynomial import polyvander
from numpy.polynomial.hermite import hermvander
from enum import Enum
import numpy as np



class BasisType(Enum):
    POLYNOMIAL = "polynomial" # Standard monomials (1, x, x^2...)
    LAGUERRE = "laguerre"     # Laguerre polynomials (Industry standard for LSMC)
    HERMITE = "hermite"       # Hermite polynomials (Good for normal distributions)

class RegressionType(Enum):
    OLS = "ols"               # Ordinary Least Squares
    RIDGE = "ridge"           # Tikhonov regularization (L2)

@dataclass
class LSMCConfig:
    n_paths: int = 50000
    n_steps: int = 100
    seed: int = 42
    
    # ML / Regression Hyperparameters
    basis_type: BasisType = BasisType.LAGUERRE
    degree: int = 3
    regression_type: RegressionType = RegressionType.OLS
    ridge_alpha: float = 1.0  # Only used if RegressionType is RIDGE

class LSMCEngine:
    def __init__(self, config: LSMCConfig):
        self.config = config
        self.rng = np.random.default_rng(self.config.seed)

    # def _generate_gbm_paths(self, S0: float, r: float, sigma: float, T: float, q: float) -> np.ndarray:
    #     """Generates standard risk-neutral GBM paths."""
    #     dt = T / self.config.n_steps
    #     drift = (r - q - 0.5 * sigma**2) * dt
    #     diffusion = sigma * np.sqrt(dt)
    #     Z = self.rng.standard_normal((self.config.n_paths, self.config.n_steps))
        
    #     paths = np.zeros((self.config.n_paths, self.config.n_steps + 1))
    #     paths[:, 0] = S0
    #     paths[:, 1:] = S0 * np.exp(np.cumsum(drift + diffusion * Z, axis=1))
    #     return paths
    def _generate_gbm_paths(self, S0: float, r: float, sigma: float, T: float, q: float, correlation_matrix: np.ndarray = None) -> np.ndarray:
        """Generates standard risk-neutral GBM paths with Antithetic Variates, supporting multi-asset."""
        dt = T / self.config.n_steps
        
        is_multi_asset = isinstance(S0, np.ndarray) and S0.size > 1
        
        # Generate HALF the required random numbers
        half_paths = self.config.n_paths // 2
        
        if is_multi_asset:
            d = len(S0)
            Z = self.rng.standard_normal((half_paths, self.config.n_steps, d))
            
            # Mirror the shocks to create the other half (Antithetic Variates)
            Z = np.concatenate((Z, -Z), axis=0)
            
            # Ensure we return exactly n_paths in case n_paths was an odd number
            if Z.shape[0] < self.config.n_paths:
                extra_Z = self.rng.standard_normal((1, self.config.n_steps, d))
                Z = np.concatenate((Z, extra_Z), axis=0)
            
            if correlation_matrix is not None:
                L = np.linalg.cholesky(correlation_matrix)
                Z = Z @ L.T
                
            drift = (r - q - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt)
            
            paths = np.zeros((self.config.n_paths, self.config.n_steps + 1, d))
            paths[:, 0, :] = S0
            paths[:, 1:, :] = S0 * np.exp(np.cumsum(drift + diffusion * Z, axis=1))
            return paths
        else:
            Z = self.rng.standard_normal((half_paths, self.config.n_steps))
            Z = np.concatenate((Z, -Z), axis=0)
            
            if Z.shape[0] < self.config.n_paths:
                extra_Z = self.rng.standard_normal((1, self.config.n_steps))
                Z = np.concatenate((Z, extra_Z), axis=0)
                
            drift = (r - q - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt)
                
            paths = np.zeros((self.config.n_paths, self.config.n_steps + 1))
            paths[:, 0] = S0
            paths[:, 1:] = S0 * np.exp(np.cumsum(drift + diffusion * Z, axis=1))
            return paths

    def _payoff(self, spot_prices: np.ndarray, strike: float, option_type: OptionType) -> np.ndarray:
        if option_type == OptionType.PUT:
            return np.maximum(strike - spot_prices, 0.0)
        elif option_type == OptionType.CALL:
            return np.maximum(spot_prices - strike, 0.0)
        raise ValueError(f"Unknown option type: {option_type}")

    def _build_basis(self, X: np.ndarray) -> np.ndarray:
        """Generates the design matrix based on the configured basis type."""
        if self.config.basis_type == BasisType.LAGUERRE:
            return lagvander(X, self.config.degree)
        elif self.config.basis_type == BasisType.POLYNOMIAL:
            return polyvander(X, self.config.degree)
        elif self.config.basis_type == BasisType.HERMITE:
            return hermvander(X, self.config.degree)
        raise ValueError(f"Unsupported basis type: {self.config.basis_type}")

    def _regress(self, X_mat: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Fits the regression model based on the configured regression type."""
        if self.config.regression_type == RegressionType.OLS:
            return np.linalg.lstsq(X_mat, Y, rcond=None)[0]
        
        elif self.config.regression_type == RegressionType.RIDGE:
            # Ridge regression: (X^T X + alpha * I)^-1 X^T Y
            # Using raw numpy to avoid forcing a scikit-learn dependency on the toolkit
            I = np.eye(X_mat.shape[1])
            I[0, 0] = 0.0  # Conventionally, we do not penalize the intercept
            penalty = self.config.ridge_alpha * I
            return np.linalg.solve(X_mat.T @ X_mat + penalty, X_mat.T @ Y)
            
        raise ValueError(f"Unsupported regression type: {self.config.regression_type}")

    def price(self, instrument: Option, market: MarketData) -> OptionPriceResult:
        is_multi = isinstance(market, MultiAssetMarketData)
        S0, r, sigma = market.spot, market.rate, market.volatility
        K, T = instrument.strike, instrument.maturity
        q = market.dividend_yield
        dt = T / self.config.n_steps
        df = np.exp(-r * dt)

        # 1. Path Generation
        if is_multi:
            paths = self._generate_gbm_paths(S0, r, sigma, T, q, market.correlation_matrix)
        else:
            paths = self._generate_gbm_paths(S0, r, sigma, T, q)

        # Handle Basket Option Aggregation
        if getattr(instrument, 'basket_type', None) is not None and is_multi:
            weights = instrument.weights
            if weights is None:
                weights = np.ones(len(S0)) / len(S0)
                
            if instrument.basket_type == BasketType.AVERAGE:
                paths = np.average(paths, axis=2, weights=weights)
            elif instrument.basket_type == BasketType.MAX:
                paths = np.max(paths * weights, axis=2)
            elif instrument.basket_type == BasketType.MIN:
                paths = np.min(paths * weights, axis=2)
                
            S0 = paths[0, 0] # Override scalar S0 for correct T=0 check

        # If it's a European option, we can skip the backward induction and directly compute the discounted payoff
        if instrument.style == OptionStyle.EUROPEAN:
            terminal_payoffs = self._payoff(paths[:, -1], K, instrument.option_type)
            return OptionPriceResult(
                price=float(np.mean(terminal_payoffs) * np.exp(-r * T)),
                boundary_times=np.array([]),
                boundary_spots=np.array([])
            )
        cash_flows = self._payoff(paths[:, -1], K, instrument.option_type)
        
        # Boundary tracking arrays
        # boundary_spots = np.zeros(self.config.n_steps + 1)
        boundary_spots = np.full(self.config.n_steps + 1, np.nan)
        boundary_times = np.linspace(0, T, self.config.n_steps + 1)
        boundary_spots[-1] = K 

        # High-resolution deterministic grid for boundary root-finding
        grid_resolution = 5000 
        if instrument.option_type == OptionType.PUT:
            spot_grid = np.linspace(1e-4, K, grid_resolution)
        else:
            spot_grid = np.linspace(K, 100*K, grid_resolution)

        
        payoff_grid = self._payoff(spot_grid, K, instrument.option_type)
        basis_grid = self._build_basis(spot_grid / K) # Pre-compute basis for the grid



        # 2. Backward Induction
        for t in range(self.config.n_steps - 1, 0, -1):
            cash_flows = cash_flows * df
            current_payoff = self._payoff(paths[:, t], K, instrument.option_type)
            itm = current_payoff > 0

            if np.sum(itm) > self.config.degree + 1:
                # Normalize spot prices (X/K) for numerical stability before creating polynomials
                X_itm = paths[itm, t] / K 
                Y_itm = cash_flows[itm]

                # Generate design matrix and solve regression
                basis_matrix = self._build_basis(X_itm)
                coeffs = self._regress(basis_matrix, Y_itm)

                # Exercise decision
                continuation_value = basis_matrix @ coeffs
                exercise = current_payoff[itm] > continuation_value
                cash_flows[itm] = np.where(exercise, current_payoff[itm], cash_flows[itm])

                # --- Boundary Extraction ---

                cont_grid = basis_grid @ coeffs
                exercise_zone = (payoff_grid > cont_grid) & (payoff_grid > 0)
                
                if np.any(exercise_zone):
                    if instrument.option_type == OptionType.PUT:
                        boundary_spots[t] = spot_grid[exercise_zone].max()
                    else:
                        boundary_spots[t] = spot_grid[exercise_zone].min()
                else:
                    boundary_spots[t] = np.nan

        # 3. Time T=0 logic
        cash_flows = cash_flows * df
        expected_pv = np.mean(cash_flows)
        immediate_exercise_t0 = self._payoff(np.array([S0]), K, instrument.option_type)[0]
        
        final_price = max(float(expected_pv), immediate_exercise_t0)
        boundary_spots[0] = S0 if immediate_exercise_t0 > expected_pv else np.nan

        return OptionPriceResult(
            price=final_price,
            boundary_times=boundary_times,
            boundary_spots=boundary_spots
        )