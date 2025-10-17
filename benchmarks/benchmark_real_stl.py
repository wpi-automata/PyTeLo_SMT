"""
Benchmark REAL STL formulas using PyTeLo
Compares Gurobi vs PuLP solvers (SCIP, HiGHS, CBC) on actual STL problems
"""

import sys
import os
# Add parent directory and stl directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stl'))

import time
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Try importing PyTeLo components
print("="*70)
print("Checking PyTeLo availability...")
print("="*70)

PYTELO_AVAILABLE = False
GUROBI_AVAILABLE = False
PULP_AVAILABLE = False

try:
    import stl as stl_module
    print("[OK] Imported PyTeLo STL module")
    print("     Location: {}".format(stl_module.__file__))
    PYTELO_AVAILABLE = True
except ImportError as e:
    print("[FAIL] Cannot import PyTeLo STL: {}".format(e))

try:
    import stl2milp
    print("[OK] Imported stl2milp (Gurobi encoder)")
    GUROBI_AVAILABLE = True
except ImportError as e:
    print("[WARNING] Cannot import stl2milp: {}".format(e))

try:
    import stl2milp_pulp
    print("[OK] Imported stl2milp_pulp (PuLP encoder)")
    PULP_AVAILABLE = True
except ImportError as e:
    print("[WARNING] Cannot import stl2milp_pulp: {}".format(e))

try:
    import gurobipy
    print("[OK] Gurobi Python package available")
except ImportError:
    print("[WARNING] Gurobi not available")

try:
    from pulp import *
    print("[OK] PuLP available")
except ImportError:
    print("[WARNING] PuLP not available")

print()


# Real STL formulas from your meeting
STL_FORMULAS = [
    {
        'id': '1',
        'formula': 'G[0,1] x >= 3',
        'description': 'Always x >= 3 during time [0,1]',
        'variables': ['x'],
        'bounds': {'x': [-100, 100]},
        'time_horizon': 10
    },
    {
        'id': '2',
        'formula': '(x > 10) && F[0, 2] y > 2 || G[1, 6] a > 8',
        'description': 'Complex formula with multiple operators',
        'variables': ['x', 'y', 'a'],
        'bounds': {'x': [-100, 100], 'y': [-100, 100], 'a': [-100, 100]},
        'time_horizon': 10
    },
    {
        'id': '3',
        'formula': 'G[2,4] F[1,3] (x>=3)',
        'description': 'Nested temporal operators',
        'variables': ['x'],
        'bounds': {'x': [-100, 100]},
        'time_horizon': 10
    },
    {
        'id': '5',
        'formula': '(x < 10) && F[0, 2] y > 2 || x >= 3',
        'description': 'Mixed logical and temporal',
        'variables': ['x', 'y'],
        'bounds': {'x': [-100, 100], 'y': [-100, 100]},
        'time_horizon': 10
    },
    {
        'id': '1complex w/o dyn',
        'formula': '(x <= 10) && F[0, 2] x > 2 && G[1, 6] (x < 8) && G[1,6] (x > 3)',
        'description': 'Complex without dynamics',
        'variables': ['x'],
        'bounds': {'x': [-100, 100]},
        'time_horizon': 10
    },
]


def parse_stl_formula(formula_str):
    """
    Parse STL formula string using PyTeLo
    
    NOTE: This is a placeholder - actual parsing depends on PyTeLo's API
    which we need to discover by looking at examples
    """
    if not PYTELO_AVAILABLE:
        return None
    
    try:
        # This is how PyTeLo MIGHT parse formulas
        # We need to check examples to see the real API
        
        # Option 1: If there's a parse function in stl module
        if hasattr(stl_module, 'parse'):
            ast = stl_module.parse(formula_str)
            return ast
        
        # Option 2: If there's a Formula class
        if hasattr(stl_module, 'Formula'):
            ast = stl_module.Formula(formula_str)
            return ast
        
        # For now, just return the string
        print("     [WARNING] Don't know how to parse yet - need to check PyTeLo examples")
        return formula_str
        
    except Exception as e:
        print("     [FAIL] Parse error: {}".format(e))
        return None


def solve_with_gurobi_placeholder(formula_dict, num_runs=3):
    """
    Placeholder for solving with Gurobi via PyTeLo's stl2milp
    
    TODO: Implement actual solving once we understand stl2milp API
    """
    times = []
    
    for run in range(num_runs):
        # Simulate solving
        # In reality, you would:
        # 1. Parse formula to AST
        # 2. Create stl2milp encoder
        # 3. Add variable bounds
        # 4. Call optimize()
        
        start = time()
        elapsed = time() - start
        times.append(elapsed)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'status': 'PLACEHOLDER - Not yet implemented',
        'solver': 'Gurobi'
    }


def solve_with_pulp_placeholder(formula_dict, solver_name, num_runs=3):
    """
    Placeholder for solving with PuLP
    
    TODO: Implement once stl2milp_pulp is complete
    """
    times = []
    
    for run in range(num_runs):
        start = time()
        elapsed = time() - start
        times.append(elapsed)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'status': 'PLACEHOLDER - Not yet implemented',
        'solver': solver_name
    }


def demonstrate_parsing():
    """
    Demonstrate that we can at least parse the formulas
    This is the first step before solving
    """
    if not PYTELO_AVAILABLE:
        print("Cannot demonstrate - PyTeLo not available")
        return
    
    print("\n" + "="*70)
    print("DEMONSTRATION: Formula Parsing")
    print("="*70)
    
    # Check what PyTeLo actually provides
    print("\nPyTeLo STL module contents:")
    items = [x for x in dir(stl_module) if not x.startswith('_')]
    for i, item in enumerate(items[:20], 1):
        obj = getattr(stl_module, item)
        obj_type = type(obj).__name__
        print("  {:2}. {:30} ({})".format(i, item, obj_type))
    
    print("\n" + "-"*70)
    print("Testing formula parsing:")
    print("-"*70)
    
    for formula_dict in STL_FORMULAS[:3]:  # Test first 3
        print("\nFormula {}: {}".format(formula_dict['id'], formula_dict['formula']))
        print("  Description: {}".format(formula_dict['description']))
        
        ast = parse_stl_formula(formula_dict['formula'])
        if ast:
            print("  [OK] Parsed successfully")
            print("    Type: {}".format(type(ast)))
        else:
            print("  [FAIL] Parsing not yet implemented")


def benchmark_formula(formula_dict, solvers, num_runs=3):
    """
    Benchmark a single formula across all solvers
    """
    results = []
    
    print("\nFormula {}: {}".format(formula_dict['id'], formula_dict['formula']))
    print("  {}".format(formula_dict['description']))
    
    for solver_name in solvers:
        print("  [{:10}] ".format(solver_name), end='', flush=True)
        
        if solver_name == 'Gurobi':
            result = solve_with_gurobi_placeholder(formula_dict, num_runs)
        else:
            result = solve_with_pulp_placeholder(formula_dict, solver_name, num_runs)
        
        result['formula_id'] = formula_dict['id']
        results.append(result)
        
        print("{:.4f}s (+/-{:.4f}) [{}]".format(
            result['mean_time'], 
            result['std_time'], 
            result['status']
        ))
    
    return results


def create_comparison_plot(df, filename='stl_comparison.png'):
    """Create comparison plot"""
    df_valid = df[df['mean_time'].notna()]
    
    if df_valid.empty:
        print("No valid data to plot")
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
               color=colors.get(solver, 'C{}'.format(i)),
               capsize=5,
               alpha=0.8)
    
    ax.set_xlabel('STL Formula', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average CPU Time (s)', fontsize=12, fontweight='bold')
    ax.set_title('STL Solver Comparison (Lower is better)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(formulas, rotation=15, ha='right')
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print("\n[OK] Plot saved to: {}".format(filename))


def main():
    """Main benchmark"""
    
    print("\n" + "="*70)
    print("STL Formula Benchmark - Real PyTeLo Integration")
    print("="*70)
    print()
    
    # First, demonstrate what we can do
    if PYTELO_AVAILABLE:
        demonstrate_parsing()
    else:
        print("[ERROR] PyTeLo not available!")
        print("\nMake sure:")
        print("1. You're in the PyTeLo root directory")
        print("2. Parser files exist in stl/ directory")
        print("3. Run: python test_import_now.py")
        return
    
    print("\n" + "="*70)
    print("BENCHMARK - Using Placeholder Solvers")
    print("="*70)
    print()
    print("NOTE: This is a DEMONSTRATION with placeholder timing.")
    print("      Real solving requires:")
    print("      1. Understanding stl2milp API")
    print("      2. Completing stl2milp_pulp implementation")
    print()
    
    # Determine which solvers to test
    solvers = []
    if GUROBI_AVAILABLE:
        solvers.append('Gurobi')
    solvers.extend(['SCIP', 'HiGHS', 'CBC'])  # PuLP solvers
    
    print("Testing solvers: {}".format(', '.join(solvers)))
    print("Formulas: {}".format(len(STL_FORMULAS)))
    print("Runs per formula: 3")
    print()
    
    # Run benchmarks
    all_results = []
    for formula in STL_FORMULAS:
        results = benchmark_formula(formula, solvers, num_runs=3)
        all_results.extend(results)
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save results
    df.to_csv('stl_benchmark_results.csv', index=False)
    print("\n[OK] Results saved to: stl_benchmark_results.csv")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    pivot = df.pivot_table(
        values='mean_time',
        index='formula_id',
        columns='solver',
        aggfunc='mean'
    )
    print("\nSolve Times (seconds):")
    print(pivot.to_string())
    
    # Create plot
    create_comparison_plot(df, 'stl_comparison.png')
    
    print("\n" + "="*70)
    print("NEXT STEPS TO MAKE THIS REAL:")
    print("="*70)
    print("1. Look at PyTeLo examples to see how to use stl2milp")
    print("2. Implement solve_with_gurobi() using real stl2milp")
    print("3. Complete stl2milp_pulp.py to translate AST to PuLP")
    print("4. Implement solve_with_pulp() using stl2milp_pulp")
    print("5. Re-run this benchmark with real solving")
    print()


if __name__ == '__main__':
    main()