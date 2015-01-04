"""
Simplify nonlinear problems by working with their linear subproblems.

"""
import fluxtools
import fluxtools.nlcm as nlcm
import glpk # from pyglpk
from fluxtools.functions import Linear
import numpy as np
from fluxtools.utilities.structure import list_fixed_variables

class LinearProgrammingFailure(Exception):
    pass

def pyglpk_linear_subproblem(model):
    lp = glpk.LPX()
    lp.obj.maximize = False
    constraint_variables = set()
    for name, g in model.constraints.items():
        if isinstance(g, Linear):
            for v in g.variables:
                if v not in constraint_variables:
                    lp.cols.add(1)
                    lp.cols[-1].name = v
                    if v in model.parameters:
                        bounds = model.parameters[v]
                    else:
                        bounds = model.get_bounds(v)
                    lp.cols[-1].bounds = bounds
                    constraint_variables.add(v)
            lp.rows.add(1)
            lp.rows[-1].name = name
            matrix = [(v, float(c)) for v,c in g.coefficients.iteritems()]
            lp.rows[-1].matrix = matrix

            lb, ub = model.get_bounds(name)
            if lb is not None and lb == ub:
                lp.rows[-1].bounds = lb
            else:
                lp.rows[-1].bounds = lb, ub

    return lp

linear_subproblem = pyglpk_linear_subproblem

def extremize(lp, variable):
    """ Overwrites objective function, leaving it zero everywhere. """
    lp.obj[:] = 0 
    lp.obj[variable] = 1

    lp.obj.maximize = False
    lp.simplex()
    if lp.status == 'opt':
        min_v = lp.obj.value
    elif lp.status == 'unbnd': 
        min_v = None
    else: 
        raise LinearProgrammingFailure()

    lp.obj.maximize = True
    lp.simplex()
    if lp.status == 'opt':
        max_v = lp.obj.value
    elif lp.status == 'unbnd': 
        max_v = None
    else: 
        raise LinearProgrammingFailure()

    lp.obj[:] = 0
    return (min_v, max_v)
    
def inside(bounds1, bounds2, tolerance=1e-6):
    """ True if bounds1 contain bounds2 to within tolerance. """
    min1, max1 = bounds1
    min2, max2 = bounds2
    min_ok = (min1 is None or min1 - tolerance <= min2)
    max_ok = (max1 is None or max1 + tolerance >= max2)
    return min_ok and max_ok

def fva(lp):
    return {col.name: extremize(lp, col.name) for col in lp.cols}

def compare_fva(fva1, fva2, tolerance=1e-6, subset_ok=True):
    if not subset_ok:
        assert fva1.keys() == fva2.keys()
    bad_keys = []
    for k, extrema1 in fva1.iteritems():
        if k not in fva2:
            continue
        extrema2 = fva2[k]
        for i in (0,1):
            bad = ((extrema1[i] is None and extrema2[i] is not None) or
                   (extrema1[i] is not None and extrema2[i] is None) or
                   np.abs(extrema1[i] - extrema2[i]) > tolerance)
            if bad:
                bad_keys.append(k)
                break
    return bad_keys

def relax_redundant_variable_bounds(lp, ignore_fixed=True):
    relaxed_variables = []
    decisions = {}
    for col in lp.cols:
        old_bounds = col.bounds
        if ignore_fixed and (old_bounds[0] == old_bounds[1]):
            continue
        col.bounds = (None, None)
        empirical_bounds = extremize(lp, col.name)
        redundant = inside(old_bounds, empirical_bounds)
        decisions[col.name] = (old_bounds, empirical_bounds, redundant)
        if redundant:
            if old_bounds != (None, None):
                relaxed_variables.append(col.name)
        else:
            col.bounds = old_bounds
    return relaxed_variables, decisions



def pop_redundant(model, threshold=1e-6):
    constraints = [g for g in model.constraints if 
                   isinstance(g, fluxtools.functions.Linear)]
    variables = list({v for g in constraints for v in g.variables})
    variable_index = {v:i for i,v in enumerate(variables)}
    n_con = len(constraints)
    n_var = len(variables)
    J = np.zeros((n_var, n_con))
    for j,g in enumerate(constraints):
        for var, c in g.coefficients.iteritems():
            i = variable_index[var]
            J[i,j] = c
    r = np.diag(np.linalg.qr(J, mode='r'))
    l = []
    for g, r_entry in zip(constraints,r):
        if np.abs(r_entry)<threshold:
            l.append(g.name)
            model.set_bound(g.name, None)
    return l, r, J, constraints


def simplify(nonlinear_model, tolerance=1e-6, relax_offset=None):
    """Relax bounds to simplify a problem, not changing its feasible region.

    The original model is unchanged.

    If relax_offset is None, redundant upper and lower bounds on variables
    will be relaxed completely. Otherwise, a variable determined to have 
    upper and lower bounds (a,b) lying inside its explicitly specified bounds
    (c,d) will have its bounds adjusted to (a-relax_offset, b+relax_offset)
    in the theory that this will improve (or at least perturb) optimizer
    performance. Note, though described as a relaxation, this may in fact
    tighten the bounds, though not in a way that will have any effect 
    in the feasible space.

    Returns:
    simplified_model - a copy with some constraints removed, some 
        variables removed or turned to parameters, and new bounds set. 
    details - a tuple (fixed_variable_values, relaxed_variables, decisions, 
        defunct_constraints, redundant_equalities) giving information 
        about variables/constraints acted on at different steps in 
        the process.

    """
    new_model = nonlinear_model.copy()
    # necessary?
    new_model.compile()
    lp0 = linear_subproblem(new_model)
    fva0 = fva(lp0)

    # First identify all variables which are constrained to be effectively
    # zero and make that constraint explicit. When IPOPT's 
    # fixed_variable_treatment make_parameter option is set and respected,
    # this ensures the upper/lower bound multipliers for these variables
    # will be zero.
    fixed_variables = []
    fixed_variable_values = {}
    for variable, (lb, ub) in fva0.iteritems():
        if lb is None or ub is None:
            continue
        # Treat variables fixed to zero specially, because there
        # are many of them and we don't want to average their
        # numerical upper/lower bounds and have the 
        # problem littered with parameters set to ~1e-12
        if np.abs(lb) < 0.5*tolerance and np.abs(ub) < 0.5*tolerance:
            fixed_variables.append(variable)
            fixed_variable_values[variable] = 0.
            new_model.set_bound(variable, 0.)
        elif np.abs(ub-lb) < tolerance:
            value = 0.5*(ub+lb)
            fixed_variables.append(variable)
            fixed_variable_values[variable] = value
            new_model.set_bound(variable, value)
    fixed_variable_set = set(fixed_variables)
            
    # Next, identify cases where upper/lower bounds on variables 
    # are redundant (because the action of other constraints already
    # forces the variables to take values within those bounds), 
    # and remove the explicit bounds on those variables, 
    # (except cases where the variable is explicitly fixed to a particular
    # value, such as those fixed in the previous step).
    lp1 = linear_subproblem(new_model)
    relaxed_variables, decisions = relax_redundant_variable_bounds(
        lp1,
        ignore_fixed=True
    )
    if relax_offset is None:
        for v in relaxed_variables:
            new_model.set_bound(v, None)
    else:
        for v in relaxed_variables:
            empirical_lb, empirical_ub = decisions[v][1]
            new_model.set_bound(v, (empirical_lb - relax_offset,
                                    empirical_ub + relax_offset))
    # Note we could add a step here to deal with redundant linear
    # inequalities in general in the very same way, but there are
    # typically few such inequalities in our models, so I haven't,
    # yet.

    # Then, completely remove any constraints dependent only on fixed
    # variables and non-optimized parameters of the problem, checking
    # in passing that any nonlinear constraints removed this way are
    # feasible
    effective_parameters = new_model.parameters.copy()
    effective_parameters.update(fixed_variable_values)
    defunct_constraints = []
    for key, g in new_model.constraints.items():
        if g.variables.issubset(effective_parameters):
            lower_bound, upper_bound = new_model.get_bounds(key)
            value = eval(g.code(), nlcm._common_namespace,
                         effective_parameters)
            assert lower_bound is None or lower_bound - value < tolerance
            assert upper_bound is None or value - upper_bound < tolerance
            defunct_constraints.append(key)
            new_model.remove_constraint(key)

    # Find linear equalities which are redundant in the reduced 
    # space of nonfixed variables, and eliminate them. This is 
    # independent of the last step so their ordering is abitrary.
    redundant_equalities = []    
    equalities = [g for key,g in new_model.constraints.items() if 
                  isinstance(g, Linear) 
                  and new_model.get_lower_bound(key) is not None and
                  new_model.get_lower_bound(key) == 
                  new_model.get_upper_bound(key)]
    eq_variables = list({v for g in equalities for v in g.variables if
                         v not in fixed_variable_set})
    eq_variable_index = {v:i for i,v in enumerate(eq_variables)}
    n_eq = len(equalities)
    n_eq_var = len(eq_variables)
    J_eq = np.zeros((n_eq_var, n_eq))
    for j,g in enumerate(equalities):
        for var, c in g.coefficients.iteritems():
            if var not in fixed_variable_set:
                i = eq_variable_index[var]
                J_eq[i,j] = c
    r_eq = np.diag(np.linalg.qr(J_eq, mode='r'))
    for g, r_entry in zip(equalities,r_eq):
        if np.abs(r_entry) < tolerance:
            redundant_equalities.append(g.name)
            new_model.remove_constraint(g.name)

    # Recompile the model, determine which fixed variables are still
    # used, and add them as parameters
    new_model.compile()
    for variable, value in fixed_variable_values.iteritems():
        if variable in new_model.variables:
            new_model.parameters[variable] = value
    new_model.compile()

    return new_model, (fixed_variable_values, relaxed_variables, decisions, 
            defunct_constraints, redundant_equalities)
    
