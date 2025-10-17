"""
Benchmark script to compare solver performance
Tests different solvers on STL planning problems
Includes visualization like the meeting plot
"""

import sys
import os
from time import time # as time_module # <-- RENAME to avoid shadowing
import pandas as pd
import numpy as np
from datetime import datetime
from pulp import *
import matplotlib.pyplot as plt


# Benchmark formulas (from your meeting notes)
TEST_PROBLEMS = [
    {
        'name': '1',
        'description': 'G[0,1] x >= 3',
        'complexity': 'simple'
    },
    {
        'name': '2',
        'description': '(x > 10) && F[0, 2] y > 2 || G[1, 6] a > 8',
        'complexity': 'simple'
    },
    {
        'name': '3',
        'description': 'G[2,4] F[1,3] (x>=3)',
        'complexity': 'simple'
    },
    {
        'name': '5',
        'description': '(x < 10) && F[0, 2] y > 2 || x >= 3',
        'complexity': 'simple'
    },
    {
        'name': 'dynamics1',
        'description': 'G[1,5] x > 5 with dynamics',
        'complexity': 'with_dynamics'
    },
    {
        'name': '1complex w/o dyn',
        'description': '(x <= 10) && F[0, 2] x > 2 && G[1, 6] (x < 8) && G[1,6] (x > 3)',
        'complexity': 'complex_no_dyn'
    },
    {
        'name': '2complex w/ dynamics',
        'description': '(x <= 10) && F[0, 2] x > 2 && G[1, 6] (x < 8) with dynamics',
        'complexity': 'complex_with_dyn'
    }
]


def create_test_problem(problem_spec):
    """Create a test optimization problem"""
    
    prob = LpProblem(f"Test_{problem_spec['name']}", LpMaximize)
    x = []  # Initialize x FIRST, before the if statements
    
    if 'simple' in problem_spec['complexity']:
        # Simple problem: 5 variables, 10 constraints
        x = [LpVariable(f"x{i}", 0, 10, cat='Integer') for i in range(5)]
        prob += lpSum(x)
        prob += lpSum(x) <= 20
        prob += x[0] + x[1] <= 8
    
    elif 'complex' in problem_spec['complexity']:
        # Complex problem: 20 variables, 40 constraints
        x = [LpVariable(f"x{i}", 0, 10, cat='Integer') for i in range(20)]
        prob += lpSum(x)
        
        # More constraints
        for i in range(0, 18, 2):
            prob += x[i] + x[i+1] <= 12
        
        for i in range(10):
            prob += lpSum([x[j] for j in range(i, min(i+5, 20))]) >= 5
    
    # Now x exists in scope - check if it has elements
    if 'dynamics' in problem_spec['complexity'] and len(x) > 1:
        # Add dynamics constraints
        for i in range(len(x)-1):
            prob += x[i+1] >= x[i] - 2
            prob += x[i+1] <= x[i] + 2
    
    return prob


def benchmark_solver(solver_name, problem_spec, num_runs=3, timeout=60):
    """Benchmark a single solver on a problem multiple times"""
    
    times = []
    statuses = []
    
    for run in range(num_runs):
        try:
            # Create fresh problem each time
            prob = create_test_problem(problem_spec)
            
            if prob is None:
                print(f"Problem creation failed!")
                times.append(None)
                statuses.append('PROBLEM_ERROR')
                continue
            
            # Get solver - CORRECTED VERSIONS
            try:
                if solver_name == 'HiGHS':
                    # Try Python API first
                    try:
                        import highspy
                        solver = HiGHS(msg=0)
                    except:
                        solver = HiGHS_CMD(msg=0, timeLimit=timeout)
                
                elif solver_name == 'CBC':
                    solver = PULP_CBC_CMD(msg=0, timeLimit=timeout)
                
                elif solver_name == 'SCIP':
                # FIX: Try Python API first, like test_basic.py does
                    try:
                        from pyscipopt import Model
                        solver = SCIP_PY(msg=False)
                    except ImportError:
                        solver = SCIP_CMD(msg=0, timeLimit=timeout)
                
                elif solver_name == 'Gurobi':
                    solver = GUROBI_CMD(msg=0, timeLimit=timeout)
                
                elif solver_name == 'CLP':
                    solver = COIN_CMD(msg=0, maxSeconds=timeout)
                
                else:
                    print(f"Unknown solver: {solver_name}")
                    times.append(None)
                    statuses.append('UNKNOWN_SOLVER')
                    continue
            
            except Exception as e:
                print(f"Solver init error: {e}")
                times.append(None)
                statuses.append(f'INIT_ERROR')
                continue
            
            # Solve
            start = time()
            status = prob.solve(solver)
            elapsed = time() - start
            
            times.append(elapsed)
            statuses.append(LpStatus[status])
            
            # Debug output for first run
            if run == 0:
                print(f"[run {run+1}: {elapsed:.3f}s, status={LpStatus[status]}]", end=' ')
            
        except Exception as e:
            print(f"RUN ERROR: {e}", end=' ')
            times.append(None)
            statuses.append(f'RUNTIME_ERROR')
    
    # Calculate statistics
    valid_times = [t for t in times if t is not None]
    
    if valid_times:
        result = {
            'solver': solver_name,
            'problem': problem_spec['name'],
            'mean_time': np.mean(valid_times),
            'std_time': np.std(valid_times),
            'min_time': np.min(valid_times),
            'max_time': np.max(valid_times),
            'success_rate': len(valid_times) / len(times),
            'status': statuses[0] if statuses else 'ERROR'
        }
        print(f"→ {result['mean_time']:.4f}s ±{result['std_time']:.4f}")
        return result
    else:
        print(f"→ ALL RUNS FAILED (statuses: {statuses})")
        return {
            'solver': solver_name,
            'problem': problem_spec['name'],
            'mean_time': None,
            'std_time': 0,
            'min_time': None,
            'max_time': None,
            'success_rate': 0,
            'status': f'FAILED: {statuses[0] if statuses else "UNKNOWN"}'
        }

def run_benchmark_suite(solvers, num_runs=3):
    """Run full benchmark suite"""
    
    print("="*70)
    print("Solver Benchmark Suite")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Runs per problem: {num_runs}")
    print("="*70)
    print()
    
    results = []
    
    for i, problem in enumerate(TEST_PROBLEMS, 1):
        print(f"[{i}/{len(TEST_PROBLEMS)}] {problem['name']}: {problem['description']}")
        
        for solver in solvers:
            print(f"  [{solver}] ", end='', flush=True)
            
            result = benchmark_solver(solver, problem, num_runs)
            results.append(result)
            
            if result['mean_time'] is not None:
                print(f"{result['mean_time']:.4f}s (±{result['std_time']:.4f})")
            else:
                print(f"{result['status']}")
        
        print()
    
    return pd.DataFrame(results)


def create_comparison_plot(df, output_file='solver_comparison.png'):
    """
    Create grouped bar chart like the meeting image
    Compares all solvers across all problems
    """
    
    # Filter out failed runs
    df_valid = df[df['mean_time'].notna()].copy()
    
    if df_valid.empty:
        print("No valid results to plot!")
        return
    
    # Get unique solvers and problems
    solvers = df_valid['solver'].unique()
    problems = df_valid['problem'].unique()
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Bar width and positions
    bar_width = 0.8 / len(solvers)
    x = np.arange(len(problems))
    
    # Colors for each solver (you can customize these)
    colors = {
        'SMT': '#1f77b4',      # Blue
        'Gurobi': '#ff7f0e',   # Orange
        'HiGHS': '#2ca02c',    # Green
        'SCIP': '#d62728',     # Red
        'CBC': '#9467bd',      # Purple
        'CyLP': '#8c564b',     # Brown
    }
    
    # Plot bars for each solver
    for i, solver in enumerate(solvers):
        solver_data = df_valid[df_valid['solver'] == solver]
        
        # Get times in same order as problems
        times = []
        errors = []
        for problem in problems:
            problem_data = solver_data[solver_data['problem'] == problem]
            if not problem_data.empty:
                times.append(problem_data['mean_time'].values[0])
                errors.append(problem_data['std_time'].values[0])
            else:
                times.append(0)
                errors.append(0)
        
        # Plot
        offset = (i - len(solvers)/2 + 0.5) * bar_width
        bars = ax.bar(x + offset, times, bar_width, 
                     yerr=errors, 
                     label=solver,
                     color=colors.get(solver, f'C{i}'),
                     capsize=5,
                     alpha=0.8)
    
    # Customize plot
    ax.set_xlabel('Problem', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average CPU Time (s)', fontsize=12, fontweight='bold')
    ax.set_title('Solver Comparison by Problem (Lower is better)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(problems, rotation=15, ha='right')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add minor gridlines
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Also show the plot
    plt.show()


def create_summary_table(df):
    """Create summary statistics table"""
    
    print("\n" + "="*70)
    print("DETAILED RESULTS")
    print("="*70 + "\n")
    
    # Pivot table
    pivot = df.pivot_table(
        values='mean_time',
        index='problem',
        columns='solver',
        aggfunc='mean'
    )
    
    print("Average Solve Times (seconds):")
    print(pivot.to_string())
    
    print("\n" + "-"*70 + "\n")
    
    # Overall averages
    print("Overall Averages by Solver:")
    for solver in df['solver'].unique():
        solver_data = df[df['solver'] == solver]['mean_time']
        valid_data = solver_data.dropna()
        
        if len(valid_data) > 0:
            avg = valid_data.mean()
            std = valid_data.std()
            print(f"  {solver:15} {avg:.4f}s (±{std:.4f}s)")
        else:
            print(f"  {solver:15} No valid results")
    
    print()


def print_summary(df):
    """Print summary statistics"""
    
    create_summary_table(df)
    
    # Success rates
    print("-"*70 + "\n")
    print("Success Rates:")
    for solver in df['solver'].unique():
        solver_data = df[df['solver'] == solver]
        success = solver_data['success_rate'].mean() * 100
        print(f"  {solver:15} {success:.1f}%")
    
    print()


def test_solver_basic(solver_name):
    """Quick test that solver actually works"""
    try:
        prob = LpProblem("QuickTest", LpMaximize)
        x = LpVariable("x", 0, 10)
        prob += x
        prob += x <= 5
        
        if solver_name == 'HiGHS':
            try:
                import highspy
                solver = HiGHS(msg=0)
            except:
                solver = HiGHS_CMD(msg=0)
        elif solver_name == 'CBC':
            solver = PULP_CBC_CMD(msg=0)
        elif solver_name == 'SCIP':
            solver = SCIP_PY(msg=0)
        elif solver_name == 'Gurobi':
            solver = GUROBI_CMD(msg=0)
        else:
            return False, "Unknown solver"
        
        status = prob.solve(solver)
        
        if status == 1:  # Optimal
            return True, "Working"
        else:
            return False, f"Status: {LpStatus[status]}"
    
    except Exception as e:
        return False, str(e)


def main():
    """Main benchmark execution"""
    
    # Solvers to test
    solvers = [
        'CBC',
        'HiGHS',
        'SCIP',
        'Gurobi',
    ]
    
    print("="*70)
    print("Pre-flight Check")
    print("="*70)
    
    # Test each solver first
    working_solvers = []
    for solver in solvers:
        works, msg = test_solver_basic(solver)
        if works:
            print(f"✓ {solver:15} {msg}")
            working_solvers.append(solver)
        else:
            print(f"✗ {solver:15} {msg}")
    
    print()
    
    if not working_solvers:
        print("ERROR: No solvers are working!")
        print("Run test_basic.py first to diagnose the issue")
        return
    
    print(f"Testing with {len(working_solvers)} solver(s): {', '.join(working_solvers)}")
    print()
    
    # Run benchmarks
    df = run_benchmark_suite(working_solvers, num_runs=10)
    
    # Print summary
    print_summary(df)
    
    # Save raw results
    output_csv = 'benchmark_results.csv'
    df.to_csv(output_csv, index=False)
    print(f"Raw results saved to: {output_csv}")
    
    # Create visualization (only if we have valid data)
    if df['mean_time'].notna().any():
        create_comparison_plot(df, 'solver_comparison.png')
    else:
        print("\n No valid data to plot - all solvers failed")
    
    print("\n" + "="*70)
    print("Benchmark complete!")
    print("="*70)


if __name__ == '__main__':
    main()