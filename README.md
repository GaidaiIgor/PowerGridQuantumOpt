# Power Grid Optimization
Uses various solvers to solve Alternating Current Optimal Power Flow with Unit Commitment (AC-OPF-UC) problem and compare performance.
Existing solvers: 
1. SCIP, mixed integer
2. Smac (bayesian optimization) for binary variables + CasADi + IPOPT for continuous variables
3. Uniform sampling for binary variables + CasADi + IPOPT for continuous variables
4. Variational Quantum Algorithm (VQA) for binary variables + CasADi + IPOPT for continuous variables
