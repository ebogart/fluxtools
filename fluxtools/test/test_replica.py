""" Basic replica model setup and solving tests. """

import numpy as np
import fluxtools.replica as r
import fluxtools.nlcm as nlcm
import fluxtools.functions as fn
from nose.tools import assert_almost_equals
from test_nlcm import assert_close

def assert_dict_almost_equals(d1, d2, **kwargs):
    assert d1.keys() == d2.keys()
    for k,v in d1.iteritems():
        assert_almost_equals(v, d2[k], **kwargs)

class Circle_Test():
    def setup(self):
        m = nlcm.NonlinearModel()
        m.add_constraint('radius', 'x**2+y**2-radius_parameter', 0.)
        m.set_objective('agreement', '(x-datum_x)**2 + (y-datum_y)**2')
        m.parameters['datum_x'] = 1.5
        m.parameters['datum_y'] = 0.0
        m.parameters['radius_parameter'] = 1.
        m.compile()
        self.model = m

    def bounds_test(self):
        self.model.set_bound('x', (-5., 5.))
        m = r.ReplicaModel(self.model, 2)
        assert m.get_bounds('image0_x' == (-5., 5.))
        assert m.get_bounds('image1_x' == (-5., 5.))

    def simple_test(self):
        m = r.ReplicaModel(self.model, 2)
        m.set_objective('combined', 'image0_objective_value + image1_objective_value')
        assert_dict_almost_equals(m.parameters, self.model.parameters)
        m.compile()
        m.solve()
        expected_soln = {'image0_objective_value': 0.25,
                         'image0_x': 1.0,
                         'image0_y': 0.,
                         'image1_objective_value': 0.25,
                         'image1_x': 1.0,
                         'image1_y': 0.}
        assert_dict_almost_equals(expected_soln, m.soln)

    def auto_objective_test(self):
        m = r.ReplicaModel(self.model, 2)
        assert isinstance(m.objective_function, fn.Linear)
        assert_dict_almost_equals({k: float(v) for k,v in 
                                   m.objective_function.coefficients.iteritems()},
                                  {'image0_objective_value': 1,
                                   'image1_objective_value': 1})

    def split_test(self):
        m = r.ReplicaModel(self.model,2, split_parameters=('datum_x', 'datum_y'))
        expected_parameters = {'image0_datum_x': 1.5,
                               'image0_datum_y': 0.0,
                               'image1_datum_x': 1.5,
                               'image1_datum_y': 0.0,
                               'radius_parameter': 1.0}
        assert_dict_almost_equals(expected_parameters, m.parameters)
        m.set_objective('combined', 'image0_objective_value + image1_objective_value')
        m.parameters['image1_datum_x'] = 0
        m.parameters['image1_datum_y'] = 1.5
        m.parameters.pop('radius_parameter')
        m.solve()
        expected_soln = {'image0_objective_value': 0.0,
                         'image0_x': 1.5,
                         'image0_y': 0.,
                         'image1_objective_value': 0.0,
                         'image1_x': 0.,
                         'image1_y': 1.5,
                         'radius_parameter': 2.25}
        assert_dict_almost_equals(expected_soln, m.soln)
        assert_dict_almost_equals(m.soln_by_image[0], {'x': 1.5, 'y':0.})
        assert_dict_almost_equals(m.soln_by_image[1], {'x': 0., 'y':1.5})
        assert_dict_almost_equals(m.soln_global, {'radius_parameter': 2.25})
        assert_close(m.x_by_image[0], np.array((1.5, 0.)))
        assert_close(m.x_by_image[1], np.array((0., 1.5)))

    def nameless_constraint_test(self):
        self.model.constraints.get('radius').name = None
        m = r.ReplicaModel(self.model, 2)
        assert m.get_bounds('image0_radius') == (0., 0.)
