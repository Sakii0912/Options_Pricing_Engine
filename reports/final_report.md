# Final Report: Option Pricing Engine

## Project Overview
[To be filled with project summary and objectives]

## Mathematical Framework

### 1. Black-Scholes-Merton Model
#### Theory
[Explain BSM assumptions and formula]

#### Implementation Details
[Describe implementation approach, numerical methods]

#### Validation Results
[Results from BSM tests]

### 2. Binomial Tree Model
#### Theory
[Explain CRR/JR models and backward induction]

#### Implementation Details
[Describe tree construction and early exercise logic]

#### Convergence Analysis
[Convergence to BSM results]

#### Validation Results
[Results from tree tests]

### 3. Least Squares Monte Carlo (LSMC)

#### Theory
[Explain optimal stopping problem and regression approach]

#### Implementation Details
- Basis function selection: [describe options and choices made]
- Regressor types: [describe options and choices made]
- Hyperparameter tuning: [describe tuning process and results]

#### Validation Results
[Results from LSMC tests]

## Pricing Invariants Validation

### Invariant 1: American Put >= European Put
[Results and discussion]

### Invariant 2: American Call (No Dividend) = European Call
[Results and discussion]

### Invariant 3: Tree Convergence to BSM
[Convergence plots and analysis]

## Model Limitations

### General Limitations
- [List key assumptions and when they may break down]
- [Discuss edge cases]

### BSM Limitations
- European-style execution only
- Constant volatility assumption
- Perfect market assumptions

### Binomial Tree Limitations
- Bermudan approximation (discrete time)
- Computational complexity scales with number of steps
- Limited to lower-dimensional problems

### LSMC Limitations
- Monte Carlo sampling error
- Regression basis selection impact
- Curse of dimensionality for high-dimensional problems

## Performance Analysis

### Runtime Comparison
[Timing results across different engines and parameter settings]

### Accuracy Analysis
[Error metrics and comparisons]

### Runtime vs Accuracy Tradeoff
[Discussion of tradeoff curves and practical implications]

## ML Component Analysis

### Basis Function Impact
[Analysis of different basis functions on LSMC performance]

### Regressor Type Comparison
[Linear vs Neural Net regressors]

### Hyperparameter Tuning Results
[Impact of key hyperparameters]

## Key Findings

1. [Finding 1]
2. [Finding 2]
3. [Finding 3]

## How the Course Helped

### Theoretical Foundation
[Discuss how course material informed the project]

### Mathematical Concepts Applied
[List relevant course topics]

### Practical Implementation Insights
[Any practical lessons from the course]

## Future Enhancements

1. [Possible extension 1]
2. [Possible extension 2]
3. [Possible extension 3]

## References

[Key academic papers and references]

## Appendix

### A. Detailed Test Results
[Test output summary]

### B. Code Documentation
[Link to generated documentation]

### C. Computational Environment
[System specs and dependencies used]
