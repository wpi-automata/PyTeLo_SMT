# PuLP Integration for PyTeLo

This fork adds support for multiple optimization solvers via PuLP.

## What Was Added

- `stl/stl2milp_pulp.py` - PuLP-based encoder (alternative to Gurobi)
- `benchmarks/benchmark_solvers.py` - Comparison benchmarks
- `benchmarks/test_basic.py` - Quick solver test

## Installation
```bash
# Activate your virtual environment
source smt/Scripts/activate  # or venv/Scripts/activate

# Install additional packages
pip install pulp highspy cylp

# Install SCIP (optional, via conda)
conda install -c conda-forge scip pyscipopt