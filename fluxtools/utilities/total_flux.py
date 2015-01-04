"""
Tools for finding minimum-flux optimal solutions. 

"""
from fluxtools.replica import enclose
from fluxtools.functions import Linear, Function
from fluxtools.nlcm import OptimizationFailure

def add_total_flux_objective(model, reactions=None, flux_upper_bound=1e4):
    """Set up model to allow penalization of total flux.

    The model's current objective function is refashioned as a 
    constraint ('_objective_constraint') requiring the variable
    '_objective' to take an appropriate value. (Note this is
    slightly different from the approach taken in 
    fva.objective_constrained_fva.) 

    Additional constraints are added setting the auxiliary variables
    '_reverse_r' and '_forward_r' to the positive and negative parts
    of each reaction r in the argument 'reactions' (if not specified,
    [r.id for r in model.reactions] is used instead.) Then the
    variable '_total_flux' is set to the sum of all the forward and
    reverse parts (by '_total_flux_constraint'.)

    Note, all reactions are treated equivalently, regardless of
    whether they are officially reversible, etc.

    NB: despite the name, '_total_flux' is only guaranteed to equal
    the sum of absolute values of fluxes of the relevant reactions
    when the model has converged to a point which minimizes it
    (possibly as one part of a multi-term objective function,)
    because in general both the forward and reverse parts of
    r (='_forward_r - _reverse_r') may be nonzero.

    Finally, a new objective equal to 
        '_objective_factor * _objective + _flux_factor * _total_flux'
    is applied, and the parameters '_objective_factor' and
    '_flux_factor' are set to 1. and 0. respectively. The model is
    recompiled.

    This slightly cumbersome apparatus should allow the fairly
    efficient determination of the (still not necessarily unique!)
    minimal flux distribution which achieves the optimal value of the
    original objective function, by first solving with
    _objective_factor 1 and _flux_factor 0 to determine the optimal
    value v, then setting an upper bound of v on '_objective' and
    resolving with _objective_factor 0 and _flux_factor 1. Some
    coaxing may be necessary to persuade the latter problem to
    converge, eg, relaxing the bound on '_objective' by some small
    factor, or adjusting the following bounds:

    '_objective' is unbounded by default.
    '_total_flux_constraint' is bounded below by zero.
    '_forward_r', '_reverse_r', etc. are given bounds (0., flux_upper_bound).

    Nothing is returned.

    """ 
    if reactions is None:
        reactions = [r.id for r in model.reactions]

    # Transform objective function
    _objective_constraint = enclose(model.objective_function, '_objective',
                                    name = '_objective_constraint')
    
    model.constraints.set('_objective_constraint', _objective_constraint)
    model.set_bound('_objective_constraint', 0.)

    # Set up total flux variable
    total_flux_coefficients = {}
    for r in reactions:
        reverse_id = '_reverse_%s' % r
        forward_id = '_forward_%s' % r
        constraint_id = '_decomposition_%s' % r
        g = Linear({r: 1., forward_id: -1., 
                    reverse_id: 1.}, name=constraint_id)
        model.constraints.set(constraint_id, g)
        model.set_bound(constraint_id, 0.)
        model.set_bound(reverse_id, (0., flux_upper_bound))
        model.set_bound(forward_id, (0., flux_upper_bound))
        total_flux_coefficients.update({reverse_id: 1., 
                                            forward_id: 1.})
    total_flux_coefficients['_total_flux'] = -1.
    g = Linear(total_flux_coefficients, name='_total_flux_constraint')
    model.constraints.set('_total_flux_constraint', g)
    model.set_bound('_total_flux_constraint', 0.)
    model.set_bound('_total_flux', (0., None))

    # Set new objective
    obj_math = '_objective_factor * _objective + _flux_factor * _total_flux'
    model.set_objective('_combined_objective', obj_math)
    model.parameters['_objective_factor'] = 1.
    model.parameters['_flux_factor'] = 0.

    # Recompile
    model.compile()

def minimum_flux_solve(model, x0=None, offset=0., **kwargs):
    """ Find the minimum-flux optimum of a model with total flux variables.

    The model should have previously been given a hybrid total-flux objective
    function by add_total_flux_objective().

    The value of the (original) objective function is given an upper bound
    of its optimal value + offset during the flux minimization step. 
    
    The bound on the (original) objective function is relaxed, and 
    the parameters '_objective_factor' and '_flux_factor' are reset to 1.
    and 0. respectively, even if optimization fails in the flux minimization
    step. 

    """

    model.solve(x0, **kwargs)
    x1 = model.x.copy()
    model.set_bound('_objective', (None, model.obj_value+offset))
    model.parameters['_objective_factor'] = 0.
    model.parameters['_flux_factor'] = 1.
    try:
        x2 = model.solve(x1, **kwargs)
    except OptimizationFailure:
        x2 = model.solve(x0, **kwargs)
    finally:
            model.set_bound('_objective', None)
            model.parameters['_objective_factor'] = 1.
            model.parameters['_flux_factor'] = 0.

    return x2
