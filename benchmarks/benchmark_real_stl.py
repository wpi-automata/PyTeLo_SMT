"""
Benchmark REAL STL formulas using PyTeLo
Compares Gurobi vs PuLP solvers (SCIP, HiGHS, CBC) on actual STL problems
"""

import sys
import os

# Add stl directory to Python path
STL_DIR = os.path.join(os.path.dirname(__file__), '..', 'stl')
sys.path.insert(0, STL_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Import PyTeLo modules
print("="*70)
print("Loading PyTeLo modules...")
print("="*70)

# Test what's in sys.path
print(f"STL directory: {STL_DIR}")
print(f"STL dir exists: {os.path.exists(STL_DIR)}")
print()

PYTELO_AVAILABLE = False
GUROBI_ENCODER_AVAILABLE = False
PULP_ENCODER_AVAILABLE = False
GUROBI_AVAILABLE = False

try:
    # Import the stl module (stl.py)
    import stl
    print(f"[OK] Imported stl module from {stl.__file__}")
    PYTELO_AVAILABLE = True
except ImportError as e:
    print(f"[FAIL] Cannot import stl: {e}")

try:
    # Import stl2milp (Gurobi encoder)
    import stl2milp
    from stl2milp import stl2milp as GurobiEncoder
    print(f"[OK] Imported stl2milp from {stl2milp.__file__}")
    GUROBI_ENCODER_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Cannot import stl2milp: {e}")
    GUROBI_ENCODER_AVAILABLE = False

try:
    # Import stl2milp_pulp (PuLP encoder)
    import stl2milp_pulp
    from stl2milp_pulp import STL2MILPPuLP
    print(f"[OK] Imported stl2milp_pulp from {stl2milp_pulp.__file__}")
    PULP_ENCODER_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Cannot import stl2milp_pulp: {e}")
    PULP_ENCODER_AVAILABLE = False

try:
    import gurobipy
    print("[OK] Gurobi available")
    GUROBI_AVAILABLE = True
except ImportError:
    print("[WARNING] Gurobi not available")
    GUROBI_AVAILABLE = False

print()


# Test formulas - SAME AS BEFORE
STL_FORMULAS = [
    {
        'id': '1',
        'formula': 'G[0,5] x >= 3',
        'description': 'Always x >= 3 for times 0-5',
        'variables': ['x'],
        'ranges': {'x': (-100, 100)},
    },
    {
        'id': '2',
        'formula': 'F[0,5] x >= 10',
        'description': 'Eventually x >= 10 within times 0-5',
        'variables': ['x'],
        'ranges': {'x': (-100, 100)},
    },
    {
        'id': '3',
        'formula': 'G[0,3] F[0,2] x >= 5',
        'description': 'Always eventually x >= 5',
        'variables': ['x'],
        'ranges': {'x': (-100, 100)},
    },
    {
        'id': '4',
        'formula': '(x >= 5) && (y >= 3)',
        'description': 'Conjunction at time 0',
        'variables': ['x', 'y'],
        'ranges': {'x': (-100, 100), 'y': (-100, 100)},
    },
    {
        'id': '5',
        'formula': 'G[0,4] (x >= 5 && x <= 15)',
        'description': 'Always in range [5,15]',
        'variables': ['x'],
        'ranges': {'x': (-100, 100)},
    },
]


def parse_stl_formula(formula_str):
    """
    Parse STL formula string using PyTeLo's to_ast function.
    
    Args:
        formula_str: Formula string like "G[0,5] x >= 3"
    
    Returns:
        Parsed STL formula object or None
    """
    if not PYTELO_AVAILABLE:
        return None
    
    try:
        # Use PyTeLo's built-in to_ast function
        ast = stl.to_ast(formula_str)
        return ast
    except Exception as e:
        print(f"    [ERROR] Parse failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def solve_with_gurobi(formula_dict, num_runs=3):
    """
    Solve using Gurobi via stl2milp.
    
    Args:
        formula_dict: Formula specification dictionary
        num_runs: Number of runs for timing
    
    Returns:
        Results dictionary
    """
    if not GUROBI_ENCODER_AVAILABLE or not GUROBI_AVAILABLE:
        return {
            'mean_time': None,
            'std_time': 0,
            'status': 'Gurobi not available',
            'solver': 'Gurobi'
        }
    
    # Parse formula
    formula_ast = parse_stl_formula(formula_dict['formula'])
    if formula_ast is None:
        return {
            'mean_time': None,
            'std_time': 0,
            'status': 'Parse failed',
            'solver': 'Gurobi'
        }
    
    times = []
    statuses = []
    
    for run in range(num_runs):
        try:
            # Create encoder
            encoder = GurobiEncoder(
                formula_ast,
                formula_dict['ranges'],
                robust=True
            )
            
            # Translate
            encoder.translate(satisfaction=True)
            
            # Optimize
            start = time.time()
            encoder.model.optimize()
            elapsed = time.time() - start
            
            times.append(elapsed)
            status = 'Optimal' if encoder.model.status == 2 else 'Failed'
            statuses.append(status)
            
        except Exception as e:
            print(f"      [ERROR] Gurobi run {run+1}: {e}")
            statuses.append(f'Error: {str(e)[:30]}')
    
    if times:
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'status': statuses[0],
            'solver': 'Gurobi'
        }
    else:
        return {
            'mean_time': None,
            'std_time': 0,
            'status': statuses[0] if statuses else 'Failed',
            'solver': 'Gurobi'
        }


def solve_with_pulp(formula_dict, solver_name, num_runs=3):
    """
    Solve using PuLP with specified solver.
    
    Args:
        formula_dict: Formula specification
        solver_name: 'SCIP', 'HiGHS', 'CBC'
        num_runs: Number of runs
    
    Returns:
        Results dictionary
    """
    if not PULP_ENCODER_AVAILABLE:
        return {
            'mean_time': None,
            'std_time': 0,
            'status': 'PuLP encoder not available',
            'solver': solver_name
        }
    
    # Parse formula
    formula_ast = parse_stl_formula(formula_dict['formula'])
    if formula_ast is None:
        return {
            'mean_time': None,
            'std_time': 0,
            'status': 'Parse failed',
            'solver': solver_name
        }
    
    times = []
    statuses = []
    
    for run in range(num_runs):
        try:
            # Create encoder
            encoder = STL2MILPPuLP(
                formula_ast,
                formula_dict['ranges'],
                robust=True,
                solver_name=solver_name
            )
            
            # Translate
            encoder.translate(satisfaction=True)
            
            # Optimize
            result = encoder.optimize(time_limit=300, verbose=False)
            
            times.append(result['solve_time'])
            statuses.append(result['status'])
            
        except Exception as e:
            print(f"      [ERROR] {solver_name} run {run+1}: {e}")
            statuses.append(f'Error: {str(e)[:30]}')
    
    if times:
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'status': statuses[0],
            'solver': solver_name
        }
    else:
        return {
            'mean_time': None,
            'std_time': 0,
            'status': statuses[0] if statuses else 'Failed',
            'solver': solver_name
        }


def benchmark_formula(formula_dict, solvers, num_runs=3):
    """
    Benchmark single formula across all solvers.
    
    Args:
        formula_dict: Formula specification
        solvers: List of solver names
        num_runs: Runs per solver
    
    Returns:
        List of result dictionaries
    """
    results = []
    
    print(f"\n[{formula_dict['id']}] {formula_dict['formula']}")
    print(f"  {formula_dict['description']}")
    
    for solver in solvers:
        print(f"  [{solver:10}] ", end='', flush=True)
        
        if solver == 'Gurobi':
            result = solve_with_gurobi(formula_dict, num_runs)
        else:
            result = solve_with_pulp(formula_dict, solver, num_runs)
        
        result['formula_id'] = formula_dict['id']
        results.append(result)
        
        if result['mean_time'] is not None:
            print(f"{result['mean_time']:.4f}s (+/-{result['std_time']:.4f}) [{result['status']}]")
        else:
            print(f"FAILED: {result['status']}")
    
    return results


def create_comparison_plot(df, filename='stl_comparison.png'):
    """Create comparison bar chart"""
    df_valid = df[df['mean_time'].notna()].copy()
    
    if df_valid.empty:
        print("[WARNING] No valid data to plot")
        return
    
    solvers = df_valid['solver'].unique()
    formulas = df_valid['formula_id'].unique()
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    bar_width = 0.8 / len(solvers)
    x = np.arange(len(formulas))
    
    colors = {
        'Gurobi': '#ff7f0e',
        'SCIP': '#1f77b4',
        'HiGHS': '#2ca02c',
        'CBC': '#9467bd'
    }
    
    for i, solver in enumerate(solvers):
        solver_data = df_valid[df_valid['solver'] == solver]
        
        means = []
        stds = []
        for formula in formulas:
            formula_data = solver_data[solver_data['formula_id'] == formula]
            if not formula_data.empty:
                means.append(formula_data['mean_time'].values[0])
                stds.append(formula_data['std_time'].values[0])
            else:
                means.append(0)
                stds.append(0)
        
        offset = (i - len(solvers)/2 + 0.5) * bar_width
        ax.bar(x + offset, means, bar_width,
               yerr=stds,
               label=solver,
               color=colors.get(solver, f'C{i}'),
               capsize=5,
               alpha=0.8)
    
    ax.set_xlabel('STL Formula', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Solve Time (s)', fontsize=12, fontweight='bold')
    ax.set_title('STL Solver Performance Comparison', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(formulas, rotation=15, ha='right')
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Plot saved to: {filename}")


def print_summary_table(df):
    """Print summary table of results"""
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    # Create pivot table
    pivot = df.pivot_table(
        values='mean_time',
        index='formula_id',
        columns='solver',
        aggfunc='mean'
    )
    
    print("\nSolve Times (seconds):")
    print(pivot.to_string(float_format=lambda x: f'{x:.4f}'))
    
    # Calculate speedup relative to Gurobi
    if 'Gurobi' in pivot.columns:
        print("\n" + "="*70)
        print("SPEEDUP ANALYSIS (relative to Gurobi)")
        print("="*70)
        
        for solver in pivot.columns:
            if solver != 'Gurobi':
                speedup = pivot['Gurobi'] / pivot[solver]
                avg_speedup = speedup.mean()
                print(f"\n{solver}:")
                print(f"  Average speedup: {avg_speedup:.2f}x")
                print(f"  (< 1.0 means slower, > 1.0 means faster)")


def main():
    """Main benchmark execution"""
    
    print("\n" + "="*70)
    print("STL FORMULA BENCHMARK - Real PyTeLo Integration")
    print("="*70)
    print()
    
    # Check prerequisites
    if not PYTELO_AVAILABLE:
        print("[ERROR] PyTeLo not available!")
        print("\nMake sure:")
        print("1. ANTLR parsers are generated in stl/")
        print("2. You're running from PyTeLo root directory")
        return
    
    # Determine available solvers
    solvers = []
    if GUROBI_ENCODER_AVAILABLE and GUROBI_AVAILABLE:
        solvers.append('Gurobi')
    
    if PULP_ENCODER_AVAILABLE:
        solvers.extend(['SCIP', 'HiGHS', 'CBC'])
    
    if not solvers:
        print("[ERROR] No solvers available!")
        return
    
    print(f"Testing solvers: {', '.join(solvers)}")
    print(f"Formulas: {len(STL_FORMULAS)}")
    print(f"Runs per formula/solver: 3")
    print()
    
    # Run benchmarks
    all_results = []
    for formula in STL_FORMULAS:
        results = benchmark_formula(formula, solvers, num_runs=3)
        all_results.extend(results)
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save results
    csv_file = 'stl_benchmark_results.csv'
    df.to_csv(csv_file, index=False)
    print(f"\n[OK] Results saved to: {csv_file}")
    
    # Print summary
    print_summary_table(df)
    
    # Create plot
    if df['mean_time'].notna().any():
        create_comparison_plot(df, 'stl_comparison.png')
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()