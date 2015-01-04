""" Test simplification based on linear subproblems. """ 

import numpy as np
import fluxtools.simplification as s
import fluxtools.nlcm as nlcm
import fluxtools.reaction_networks as rn

class Simple_Test():
    def setup(self):
        test_network = {'tx_A': {'A': 1},
                        'R1': {'A': -1, 'B': 1, 'F': 1},
                        'R3': {'F': -1, 'B': -1, 'C': 1},
                        'R2': {'A': -1, 'F': 1, 'G': 1},
                        'sink': {'F': -1, 'G': -1}}
        test_network = rn.net_from_dict(test_network)
        full_model = nlcm.NonlinearNetworkModel('simple_yet_problematic', 
                                                test_network)
        full_model.add_constraint('equal_squares_0', 'R1**2 - R3**2', 0.)
        full_model.add_constraint('equal_squares_1', 
                                  'R1**2 - extra_var0**2 - extra_var1**2', 0.)
        full_model.set_objective('max_tx_A', '-1.0*tx_A')
        full_model.compile()
        full_model.set_bounds({'tx_A': (0., 1000.),
                               'R1': (-1000., 1000.),
                               'R3': (-1000., 1000.),
                               'R2': (-1000., 1000.),
                               'sink': (0., 1000.),
                               'extra_var0': (-5., 5.),
                               'extra_var1': (-5., 5.)})
        self.full_model = full_model
        # Note extra_var1 is just there to ensure the number of
        # variables is greater than the number of equality
        # constraints.

        # What should happen when this network is simplified:
        # Reactions R1 and R3 are blocked because only R3
        # produces/consumes metabolite C; their values should be fixed
        # to zero. Constraints _conservation_B and equal_squares_0
        # involve only R1 and R3, so they should be dropped from the
        # problem. R3 participates in no other constraints and should
        # be dropped from the problem in turn; R1 participates in
        # equal_squares_1 and should be made a parameter with value 0.
        #
        # With R1 and R3 eliminated, the constraints conserving F and
        # G become redundant, and one or the other should be
        # eliminated.
        #
        # Finally, the variable bounds on tx_A and sink are mutually
        # redundant, and one or the other should be relaxed; R2 is
        # redundant given the remaining tx_A/sink constraint and it
        # should be relaxed too.

    def test_structure(self):
        simplified_model, details = s.simplify(self.full_model)

        assert simplified_model.parameters['R1'] == 0.
        assert 'R3' not in simplified_model.variables
        assert 'R3' not in simplified_model.parameters
        assert '_conservation_B' not in simplified_model.constraints.keys()
        assert '_conservation_A' in simplified_model.constraints.keys()
        assert 'equal_squares_1' in simplified_model.constraints.keys()
        assert 'equal_squares_0' not in simplified_model.constraints.keys()
        assert len({'_conservation_F',
                    '_conservation_G'}.intersection(simplified_model.constraints.keys())) == 1
        assert simplified_model.get_bounds('R2') == (None, None)
