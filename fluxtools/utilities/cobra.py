""" Methods to extract data from models in COBRA-compliant SBML form.

Assumes the SBML model has been loaded as a fluxtools.reaction_networks.Network.

"""
import numpy as np

def get_flux_bounds(net):
    """ Extract upper and lower bounds on reaction rates. 

    Collects the UPPER_BOUND and LOWER_BOUND values read, if present,
    from the parameter lists of each reaction's kinetic law in the SBML file,
    and returns a dict {reaction1_id: (reaction1_lower, reaction1_upper), ...}

    Lower bounds of -inf and upper bounds of inf are converted to None.

    Where a reaction has neither parameter set, it is excluded from the result.
    Where only one parameter is set, the other parameter is set to None. 

    """
    bounds = {}
    for r in net.reactions:
        if hasattr(r, 'parameters') and ('UPPER_BOUND' in r.parameters or
                                         'LOWER_BOUND' in r.parameters):
            lower_bound = r.parameters.get('LOWER_BOUND', None)
            upper_bound = r.parameters.get('UPPER_BOUND', None)
            if np.isneginf(lower_bound):
                lower_bound = None
            if np.isposinf(upper_bound):
                upper_bound = None
            bounds[r.id] = (lower_bound, upper_bound)

    return bounds

def get_objective_coefficients(net, return_only_nonzero=False):
    """ Extract objective coefficients for the reactions of the network.

    Collects the OBJECTIVE_COEFFICIENT values read, if present, from 
    the parameter lists of each reaction's kinetic law in the SBML file,
    and returns {reaction1.id: objective_coefficient1, ...}

    Where no such parameter is present, a coefficient of 0. is assumed and
    returned.

    If return_only_nonzero is True, the return dict will list only the nonzero 
    coefficients.

    """
    coefficients = {}

    for r in net.reactions:
        coefficient = getattr(r, 'parameters', {}).get('OBJECTIVE_COEFFICIENT', 0.)
        coefficients[r.id] = coefficient

    if return_only_nonzero:
        return {r: c for r,c in coefficients.iteritems() if c}
    else:
        return coefficients
        
