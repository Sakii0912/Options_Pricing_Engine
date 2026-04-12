"""Global configuration and hyperparameters"""

# Default parameters for numerical methods
BINOMIAL_STEPS_DEFAULT = 100
LSMC_PATHS_DEFAULT = 10000
LSMC_STEPS_DEFAULT = 50

# Numerical tolerance
TOLERANCE = 1e-6

# Basis functions for LSMC
LSMC_BASIS_FUNCTIONS = [
    "legendre",
    "hermite",
    "laguerre"
]
