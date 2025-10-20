"""
STL to MILP encoder using PuLP instead of Gurobi
Allows using open-source solvers: SCIP, HiGHS, CBC, CLP
Drop-in replacement for stl2milp.py
"""

from pulp import *
import time

from stl import Operation, RelOperation, STLFormula


class STL2MILPPuLP:
    """
    Translate an STL formula to an MILP using PuLP backend.
    Direct port of stl2milp.py to use PuLP instead of Gurobi.
    """

    def __init__(self, formula, ranges, vtypes=None, model=None, robust=False, 
                 solver_name='SCIP'):
        """
        Args:
            formula: STL formula object (from stl.py)
            ranges: Dict mapping variable names to (min, max) bounds
                    e.g., {'x': (-100, 100), 'y': (-50, 50)}
            vtypes: Dict mapping variable names to variable types
                    'Continuous', 'Integer', 'Binary' (default: Continuous)
            model: Existing PuLP model to add to (default: create new)
            robust: If True, maximize robustness rho (default: False)
            solver_name: 'SCIP', 'HiGHS', 'CBC', 'CLP', 'GUROBI'
        """
        self.formula = formula
        self.M = 1000  # Big-M constant
        self.ranges = ranges
        self.solver_name = solver_name
        
        # Verify all formula variables have ranges
        formula_vars = set(self.formula.variables())
        range_vars = set(self.ranges.keys())
        assert formula_vars <= range_vars, \
            f"Missing ranges for variables: {formula_vars - range_vars}"
        
        # Add robustness variable range if needed
        if robust and 'rho' not in self.ranges:
            self.ranges['rho'] = (-1e6, self.M - 1)
        
        # Variable types (default: continuous)
        self.vtypes = vtypes if vtypes is not None else {}
        for v in self.ranges:
            if v not in self.vtypes:
                self.vtypes[v] = 'Continuous'
        
        # Create or use existing model
        self.model = model
        if model is None:
            self.model = LpProblem(f"STL_{formula}", LpMaximize)
        
        # Storage for variables
        self.variables = {}
        
        # Create robustness variable if needed
        if robust:
            rho_min, rho_max = self.ranges['rho']
            self.rho = LpVariable('rho', rho_min, rho_max)
            self.model += self.rho  # Objective: maximize robustness
        else:
            self.rho = 0
        
        # Map operations to encoding functions
        self._milp_call = {
            Operation.PRED: self._predicate,
            Operation.AND: self._conjunction,
            Operation.OR: self._disjunction,
            Operation.EVENT: self._eventually,
            Operation.ALWAYS: self._globally,
            Operation.UNTIL: self._until
        }
    
    def translate(self, satisfaction=True):
        """
        Translate the STL formula to MILP constraints starting at time 0.
        """
        z = self._to_milp(self.formula, t=0)
        
        if satisfaction:
            self.model += (z == 1, 'formula_satisfaction')
        
        return z
    
    def _to_milp(self, formula, t=0):
        """Recursively generate MILP encoding for formula at time t."""
        z, newly_added = self._add_formula_variable(formula, t)
        
        if newly_added:
            # Encode this formula node
            self._milp_call[formula.op](formula, z, t)
        
        return z
    
    def _add_formula_variable(self, formula, t):
        """Add a binary variable for the formula at time t."""
        if formula not in self.variables:
            self.variables[formula] = {}
        
        if t not in self.variables[formula]:
            opname = Operation.getName(formula.op)
            identifier = formula.identifier()
            name = f'{opname}_{identifier}_{t}'
            var = LpVariable(name, cat='Binary')
            self.variables[formula][t] = var
            return var, True
        
        return self.variables[formula][t], False
    
    def _add_state(self, state_name, t):
        """Add a state variable at time t."""
        if state_name not in self.variables:
            self.variables[state_name] = {}
        
        if t not in self.variables[state_name]:
            low, high = self.ranges[state_name]
            vtype = self.vtypes[state_name]
            name = f'{state_name}_{t}'
            var = LpVariable(name, low, high, cat=vtype)
            self.variables[state_name][t] = var
        
        return self.variables[state_name][t]
    
    def _predicate(self, pred, z, t):
        """Encode predicate using big-M method."""
        assert pred.op == Operation.PRED
        v = self._add_state(pred.variable, t)
        
        if pred.relation in (RelOperation.GE, RelOperation.GT):
            self.model += (v - self.M * z <= pred.threshold + self.rho,
                          f'pred_{id(pred)}_{t}_upper')
            self.model += (v + self.M * (1 - z) >= pred.threshold + self.rho,
                          f'pred_{id(pred)}_{t}_lower')
        
        elif pred.relation in (RelOperation.LE, RelOperation.LT):
            self.model += (v + self.M * z >= pred.threshold - self.rho,
                          f'pred_{id(pred)}_{t}_lower')
            self.model += (v - self.M * (1 - z) <= pred.threshold - self.rho,
                          f'pred_{id(pred)}_{t}_upper')
        else:
            raise NotImplementedError(f"Relation {pred.relation} not supported")
    
    def _conjunction(self, formula, z, t):
        """Encode conjunction (AND)."""
        assert formula.op == Operation.AND
        z_children = [self._to_milp(child, t) for child in formula.children]
        
        for z_child in z_children:
            self.model += (z <= z_child)
        
        n = len(z_children)
        self.model += (z >= 1 - n + lpSum(z_children))
    
    def _disjunction(self, formula, z, t):
        """Encode disjunction (OR)."""
        assert formula.op == Operation.OR
        z_children = [self._to_milp(child, t) for child in formula.children]
        
        for z_child in z_children:
            self.model += (z >= z_child)
        
        self.model += (z <= lpSum(z_children))
    
    def _eventually(self, formula, z, t):
        """Encode Eventually (F[a,b])."""
        assert formula.op == Operation.EVENT
        
        a, b = int(formula.low), int(formula.high)
        child = formula.child
        z_children = [self._to_milp(child, t + tau) for tau in range(a, b + 1)]
        
        for z_child in z_children:
            self.model += (z >= z_child)
        
        self.model += (z <= lpSum(z_children))
    
    def _globally(self, formula, z, t):
        """Encode Globally (G[a,b])."""
        assert formula.op == Operation.ALWAYS
        
        a, b = int(formula.low), int(formula.high)
        child = formula.child
        z_children = [self._to_milp(child, t + tau) for tau in range(a, b + 1)]
        
        for z_child in z_children:
            self.model += (z <= z_child)
        
        n = len(z_children)
        self.model += (z >= 1 - n + lpSum(z_children))
    
    def _until(self, formula, z, t):
        """Encode Until (phi U[a,b] psi) - FIXED VERSION."""
        assert formula.op == Operation.UNTIL
        
        a, b = int(formula.low), int(formula.high)
        
        # Encode children at all needed times
        z_children_left = [self._to_milp(formula.left, tau) 
                          for tau in range(t, t + b + 1)]
        z_children_right = [self._to_milp(formula.right, tau) 
                           for tau in range(t + a, t + b + 1)]
        
        z_aux = []
        phi_alw = None
        if a > 0:
            phi_alw = STLFormula(Operation.ALWAYS, child=formula.left,
                                low=t, high=t + a - 1)
        
        for k, tau in enumerate(range(t + a, t + b + 1)):
            if tau > t + a:
                phi_alw_u = STLFormula(Operation.ALWAYS, child=formula.left,
                                      low=t + a, high=tau - 1)
            else:
                phi_alw_u = formula.left
            
            children = [formula.right, phi_alw_u]
            if phi_alw is not None:
                children.append(phi_alw)
            
            phi = STLFormula(Operation.AND, children=children)
            z_conj, _ = self._add_formula_variable(phi, t)
            z_aux.append(z_conj)
            
            z_right = z_children_right[k]
            self.model += (z_conj <= z_right)
            
            # FIXED: Correct slicing
            slice_end = min(a + k + 1, len(z_children_left))
            for z_left in z_children_left[:slice_end]:
                self.model += (z_conj <= z_left)
            
            m = slice_end
            self.model += (z_conj >= 1 - m + z_right + 
                          lpSum(z_children_left[:slice_end]))
            
            self.model += (z >= z_conj)
        
        self.model += (z <= lpSum(z_aux))
    
    def optimize(self, solver=None, time_limit=300, verbose=False):
        """Solve the MILP problem."""
        if solver is None:
            solver = self._get_solver(self.solver_name, time_limit, verbose)
        
        start_time = time.time()
        status = self.model.solve(solver)
        solve_time = time.time() - start_time
        
        result = {
            'status': LpStatus[status],
            'solve_time': solve_time,
            'solver': self.solver_name,
            'objective': value(self.model.objective) if status == 1 else None,
            'solution': {}
        }
        
        if status == 1:
            for var_name in self.ranges.keys():
                if var_name == 'rho':
                    continue
                if var_name in self.variables:
                    result['solution'][var_name] = {
                        t: self.variables[var_name][t].varValue 
                        for t in self.variables[var_name]
                    }
        
        return result
    
    def _get_solver(self, name, time_limit=300, verbose=False):
        """Get PuLP solver - use Python API versions, not CMD versions"""
        msg = 1 if verbose else 0
        
        try:
            if name == 'SCIP':
                # Try Python API first (no external executable needed)
                try:
                    from pyscipopt import Model
                    # SCIP_PY uses msg as boolean
                    return SCIP_PY(msg=(msg == 1))
                except ImportError:
                    print("Warning: pyscipopt not installed, trying SCIP_CMD")
                    return SCIP_CMD(msg=msg, timeLimit=time_limit)
            
            elif name == 'HiGHS':
                # Try Python API first (no external executable needed)
                try:
                    import highspy
                    # HiGHS uses msg as integer
                    return HiGHS(msg=msg)
                except ImportError:
                    print("Warning: highspy not installed, trying HiGHS_CMD")
                    return HiGHS_CMD(msg=msg, timeLimit=time_limit)
            
            elif name == 'CBC':
                # CBC command-line (comes bundled with PuLP)
                return PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
            
            elif name == 'GUROBI':
                # Try Python API first
                try:
                    import gurobipy
                    # GUROBI uses msg as boolean
                    return GUROBI(msg=(msg == 1), timeLimit=time_limit)
                except ImportError:
                    return GUROBI_CMD(msg=msg, timeLimit=time_limit)
            
            else:
                print(f"Unknown solver {name}, using CBC")
                return PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
        
        except Exception as e:
            print(f"Error initializing {name}: {e}")
            print("Falling back to CBC")
            return PULP_CBC_CMD(msg=msg, timeLimit=time_limit)