""" Tests for fluxtools.utilities.total_flux.

Appropriate tolerances may vary.

"""
from fluxtools.nlcm import net_from_dict, NonlinearNetworkModel
from fluxtools.utilities.total_flux import add_total_flux_objective
from fluxtools.utilities.total_flux import minimum_flux_solve
from nose.tools import assert_almost_equals

# Require agreement with expected value to set number of decimal places
places = 6 

def test_abs_flux_objective():
    l1_net = net_from_dict({'input': {'A': 1},
                             'R1': {'A': -1, 'B': 1},
                             'R2': {'A': 2, 'B': -2},
                             'output': {'B': -1}})
    l1_model = NonlinearNetworkModel('l1_test', l1_net) 
    l1_model.set_objective('max_output', '-1.0*output')
    l1_model.add_constraint('pointless_constraint', 'input**2', (None, 100)) 

    add_total_flux_objective(l1_model)
    minimum_flux_solve(l1_model)
    assert_almost_equals(l1_model.obj_value, 25., places=places)
    assert_almost_equals(l1_model.soln['R2'], -5., places=places)
    assert_almost_equals(l1_model.soln['R1'], 0., places=places)
    assert_almost_equals(l1_model.soln['_objective'], -10., places=places)
