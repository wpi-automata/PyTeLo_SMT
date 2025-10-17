"""
Simple test to verify PuLP solvers work on Windows
Now correctly uses highspy when available
"""

from pulp import *
import time


def test_solver(solver_name):
    """Test if a solver works with a simple LP"""
    print(f"\nTesting {solver_name}...", end='', flush=True)
    
    try:
        # Create simple problem: maximize x subject to x <= 10
        prob = LpProblem("Test", LpMaximize)
        x = LpVariable("x", 0, None)
        prob += x
        prob += x <= 10
        
        # Get solver
        if solver_name == 'HiGHS':
            # Try HiGHS via Python API first
            try:
                import highspy
                solver = HiGHS(msg=False)  # Direct API, not command-line
            except ImportError:
                solver = HiGHS_CMD(msg=0)  # Fallback to command-line
        elif solver_name == 'CBC':
            solver = PULP_CBC_CMD(msg=0)
        elif solver_name == 'CLP':
            solver = COIN_CMD(msg=0)
        elif solver_name == 'SCIP':
            # Try SCIP via Python API first
            try:
                from pyscipopt import Model
                solver = SCIP_PY(msg=False)
            except ImportError:
                solver = SCIP_CMD(msg=0)
        elif solver_name == 'GUROBI':
            solver = GUROBI_CMD(msg=0)
        else:
            print(f" ✗ Unknown solver")
            return False
        
        # Solve
        start = time.time()
        status = prob.solve(solver)
        elapsed = time.time() - start
        
        if status == 1:  # Optimal
            print(f" ✓ Works! (x={x.varValue:.1f}, time={elapsed:.3f}s)")
            return True
        else:
            print(f" ✗ Failed (status: {LpStatus[status]})")
            return False
            
    except Exception as e:
        error_msg = str(e)
        if "cannot execute" in error_msg:
            print(f" ✗ Not installed (executable not found)")
        elif "No module named" in error_msg:
            print(f" ✗ Python package not installed")
        else:
            print(f" ✗ Error: {error_msg[:50]}")
        return False


def main():
    print("=" * 70)
    print("PuLP Solver Test (Windows)")
    print("=" * 70)
    
    solvers = ['CBC', 'CLP', 'HiGHS', 'SCIP', 'GUROBI']
    
    results = {}
    for solver in solvers:
        results[solver] = test_solver(solver)
    
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    
    working = []
    for solver, success in results.items():
        status = "✓ Available" if success else "✗ Not available"
        print(f"  {solver:10} {status}")
        if success:
            working.append(solver)
    
    print(f"\n{len(working)} solver(s) working: {', '.join(working) if working else 'NONE'}")
    
    if len(working) >= 2:
        print("\n✓ Great! You have multiple solvers for comparison.")
    elif len(working) == 1:
        print(f"\n✓ {working[0]} is working - enough to start!")
    
    print("\nNext step: Run benchmark_solvers.py")


if __name__ == '__main__':
    main()