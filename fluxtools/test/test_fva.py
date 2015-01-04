""" Tests for fluxtools.fva. 

The number of decimal places of agreement required are somewhat empirical!

"""

import fluxtools.fva as fva
import fluxtools.nlcm as nlcm
import fluxtools.reaction_networks as rn
import numpy as np 
from nose.tools import assert_almost_equals, raises

simple_model = {'tx_A': {'A_ext': -1, 'A': 1.},
                'R1': {'A': -1, 'B': 1},
                'R2': {'A': -1, 'C': 1},
                'R3': {'B': -1, 'D': 1},
                'R4': {'C': -1, 'D': 1},
                'sink': {'D': -1}}

simple_net = rn.net_from_dict(simple_model)

class SimpleNetworkFVA_Test():

    def setup(self):
        m = nlcm.NonlinearNetworkModel('simple',simple_net,
                                       external_species_ids=('A_ext',))
        m.set_objective('max_sink', '-1.0*sink')
        m.set_bound('tx_A', (0., 10.))
        # Add some nonlinear constraints so the solver does not complain--
        # set a pseudo-kinetic-law for R2
        m.add_constraint('rate_R2', 'R2-A**2', 0.) 
        m.compile()
        self.model = m

    def test_basic_fva(self):
        self.model.set_bound('R1', (0., 5.))
        result = fva.do_fva(self.model)
        expected = dict.fromkeys(simple_model, (0., 10.))
        expected['R1'] = (0., 5.)
        expected['R3'] = (0., 5.)
        expected['A'] = (0., np.sqrt(10.))
        for k in self.model.variables:
            for bound_expected, bound_found in zip(expected[k],
                                       result[k]):
                assert_almost_equals(bound_expected, bound_found, places=6)

    def test_parallel_fva(self):
        self.model.set_bound('R1', (0., 5.))
        result = fva.do_fva(self.model, n_procs=2)
        expected = dict.fromkeys(simple_model, (0., 10.))
        expected['R1'] = (0., 5.)
        expected['R3'] = (0., 5.)
        expected['A'] = (0., np.sqrt(10.))
        for k in self.model.variables:
            for bound_expected, bound_found in zip(expected[k],
                                       result[k]):
                assert_almost_equals(bound_expected, bound_found, places=6)

    def test_objective_preserved(self):
        original_objective = self.model.objective_function
        fva.do_fva(self.model, ('tx_A',))
        assert self.model.objective_function == original_objective

    def test_objective_constrained(self):
        self.model.set_bound('R1', (0., 6.))
        self.model.set_bound('R2', (0., 9.))
        self.model.solve()
        result = fva.objective_constrained_fva(self.model)
        expected = {'tx_A': (10., 10.),
                    'R1': (1., 6.),
                    'R2': (4., 9.),
                    'R3': (1., 6.),
                    'R4': (4., 9.),
                    'A': (2., 3.),
                    'sink': (10., 10.)}
        print expected
        print result
        for k in self.model.variables:
            for bound_expected, bound_found in zip(expected[k],
                                       result[k]):
                assert_almost_equals(bound_expected, bound_found, places=5)
