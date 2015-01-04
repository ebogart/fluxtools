""" Tests of the nlcm.NonlinearNetworkModel. """

import fluxtools.reaction_networks as rn
import fluxtools.nlcm as nlcm
from nose.tools import assert_almost_equals, raises

simple_model = {'tx_A': {'A_ext': -1, 'A': 1.},
                'R1': {'A': -1, 'B': 1},
                'R2': {'A': -1, 'C': 1},
                'R3': {'B': -1, 'D': 1},
                'R4': {'C': -1, 'D': 1},
                'sink': {'D': -1}}

simple_net = rn.net_from_dict(simple_model)

class Network_Test():

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

    def test_expected_solution(self):
        x = self.model.solve()
        assert_almost_equals(self.model.obj_value, -10., places=6)
        soln = self.model.soln
        assert_almost_equals(self.model.soln['R2'], self.model.soln['A']**2)
        assert_almost_equals(self.model.soln['R1'], self.model.soln['R3'])
        
    def test_species_minimum(self):
        assert self.model.get_lower_bound('A') == 0
        
    def test_constraint_names(self):
        assert not [g for g in self.model.constraints if g.name is None]

                
                
