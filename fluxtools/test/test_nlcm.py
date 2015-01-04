"""
Assorted tests of the core nonlinear optimization capabilities.

"""

from nose.tools import assert_almost_equals, raises
import numpy as np
import fluxtools.nlcm as nlcm

def assert_close(v1, v2, places=None, delta=None):
    """ Assert the L1 norm of v1-v2 is close to 0. """
    assert_almost_equals(np.max(np.abs(v1-v2)), 0., places=places, delta=delta)

def assert_nonzero_subdict(subdict, fulldict, places=None, delta=None):
    """Assert one dict is a subdict of another which is otherwise zero.

    That is, all keys of subdict are in fulldict, the dicts agree 
    on the values of those keys, and the values associated with all
    other keys of fulldict are zero.

    Here, 'agree' is as tested by nose.tools.assert_almost_equals,
    with the places and delta arguments given; values which must be
    zero, in contrast, are checked exactly.

    """
    subkeys = set(subdict)
    fullkeys = set(fulldict)
    assert subkeys.issubset(fullkeys)
    extrakeys = fullkeys - subkeys
    for k in extrakeys:
        assert fulldict[k] == 0.
    for k in subkeys:
        assert_almost_equals(subdict[k], fulldict[k], places=places,
                             delta=delta)

def assert_float_array(array, float_type=np.float):
    """ Ensure an array consists of floats. 

    Checks by default that array.dtype is np.float. 
    
    """
    assert(array.dtype == float_type)


def test_h_objective_constant_term_indexing():
    """Make sure objectives with constant second derivative terms are
    handled by Hessian calculation.

    """
    m = nlcm.NonlinearModel()
    m.add_constraint('radius', 'x**2+y**2', 1.)
    m.set_objective('agreement', '(x-datum_x)**2 + (y-datum_y)**2')
    m.parameters['datum_x'] = 1.5
    m.parameters['datum_y'] = 0.
    m.compile()
    x0 = np.array((0,0))
    h = m.eval_h(x0, [1], 1., flag=False)
    assert_close(h, np.array((4,4)))

class HS071_Test():
    """See Schittkowski `Test Examples for Nonlinear Programming
    Codes', 2009, http://www.ai7.uni-bayreuth.de/test_problem_coll.pdf; 
    Hock and Schittkowski, `Test Examples for Nonlinear Programming
    Codes', Lecture Notes in Economics and Mathematical Systems,
    Vol. 187. Springer, 1981.

    Here I have checked most of the derivatives against their 
    values at the canonical starting point (1,5,5,1), which is
    probably undesirable as its high symmetry offers many opportunities
    to obtain the correct values in the wrong way. 

    """

    def setup(self):
        m = nlcm.NonlinearModel()
        m.set_objective('hs71','x1*x4*(x1+x2+x3)+x3')
        m.add_constraint('product','x1*x2*x3*x4',(25.,None))
        m.add_constraint('radius','x1**2 + x2**2 + x3**2 + x4**2',40.)
        m.set_bounds(dict.fromkeys(['x1','x2','x3','x4'],(1.,5.)))
        m.compile()
        self.m = m
        self.x0 = np.array([1.,5.,5.,1.])

    def test_solve(self):
        # The primal variables and value at the optimal point are
        # given in the test problem collection, but only the ratio of
        # largest to smallest multipliers is given there; the values
        # here, which agree with those ratios as I understand them,
        # are the results of an apparently successful run of the
        # hs071_c example distributed with IPOPT, which ran with
        # mu_strategy adaptive, tol 1e-7, and solver ma27. (Bound
        # multipliers for inactive bounds, there on the order of
        # 1e-11, have been rounded to zero.)
        
        # The exact results will vary slightly with the choice of
        # options, especially the tolerance; at some point this test
        # should be modified to use a specified set of options. What
        # tolerance should be used _within this test_ is not
        # immediately obvious to me, but 6 decimal places seems
        # conservative without (I hope) being ridiculously
        # oversensitive.
        places = 6
        x = self.m.solve(self.x0)
        x_true = np.array((1, 4.7429994, 3.8211503, 1.3794082))
        value_true = 17.0140173
        lambda_true = np.array((-0.5522937, 0.1614686))
        zl_true = np.array((1.087871, 0,0,0))
        zu_true = np.zeros(4) 
        assert_close(x,x_true,places=places)
        assert_close(self.m.constraint_multipliers, lambda_true, places=places)
        assert_close(self.m.zl, zl_true, places=places)
        assert_close(self.m.zu, zu_true, places=places)
        assert_almost_equals(self.m.obj_value, value_true, places=places)

    def test_eval_f(self):
        assert_almost_equals(self.m.eval_f(self.x0), 16.)

    def test_eval_grad_f(self):
        grad_f_0 = np.array((12., 1., 2., 11.))
        assert_close(self.m.eval_grad_f(self.x0), grad_f_0)

    def test_eval_g(self):
        g_0 = np.array((25., 52.))
        assert_close(self.m.eval_g(self.x0), g_0)

    def test_eval_jac_g(self):
        jac_g_0 = {(0,0): 25,
                   (0,1): 5,
                   (0,2): 5,
                   (0,3): 25,
                   (1,0): 2,
                   (1,1): 10,
                   (1,2): 10, 
                   (1,3): 2}
        g_indices, v_indices = self.m.eval_jac_g(self.x0, True)
        values = self.m.eval_jac_g(self.x0, False)
        jac_g = dict(zip(zip(g_indices, v_indices), values))
        assert set(jac_g.keys()) == set(jac_g_0.keys())
        for k,v in jac_g_0.iteritems():
            assert_almost_equals(v, jac_g[k])
        
    def test_eval_h(self):
        v1, v2 = self.m.eval_h(self.x0, np.array((0., 0.)), 0., True)
        keys = zip(v1, v2)
        h_g0 = self.m.eval_h(self.x0, np.array((1., 0.)), 0., False)
        h_g1 = self.m.eval_h(self.x0, np.array((0., 1.)), 0., False)
        h_obj = self.m.eval_h(self.x0, np.array((0., 0.)), 1., False)
        true_h_g0 = {(0,1): 5, (0,2): 5., (0,3): 25., (1,2): 1., (1,3): 5., (2,3): 5.}
        true_h_g1 = {(0,0): 2., (1,1): 2., (2,2): 2., (3,3): 2.}
        true_h_obj = {(0,0): 2., (0,1): 1., (0,2): 1., (0,3): 12.,
                      (1,3): 1., (2,3): 1.}
        true_keys = set(true_h_g0.keys() + true_h_g1.keys() + true_h_obj.keys())
        print true_h_g0
        print dict(zip(keys, h_g0))
        assert set(keys) == true_keys
        assert_nonzero_subdict(true_h_g0, dict(zip(keys, h_g0)))
        assert_nonzero_subdict(true_h_g1, dict(zip(keys, h_g1)))
        assert_nonzero_subdict(true_h_obj, dict(zip(keys, h_obj)))

    @raises(nlcm.OptimizationFailure)
    def test_infeasibility_failure(self):
        self.m.set_bound('radius', -5.)
        self.m.solve()

    def test_eval_float_results(self):
        x0 = self.x0
        gf = self.m.eval_grad_f(x0)
        g = self.m.eval_g(x0)
        jac_g = self.m.eval_jac_g(x0, False)
        h_g0 = self.m.eval_h(self.x0, np.array((1., 0.)), 0., False)
        h_g1 = self.m.eval_h(self.x0, np.array((0., 1.)), 0., False)
        h_obj = self.m.eval_h(self.x0, np.array((0., 0.)), 1., False)
        for array in (gf, g, jac_g, h_g0, h_g1, h_obj):
            assert_float_array(array)
        
# TODO: tests with noninteger values?
# TODO: test that each eval_ method updates the namespace to reflect new values of x
# TODO: less symmetric example (classical photosynthesis models, or even simpler)
# TODO: sparser example, testing that only appropriate derivative entries are returned
# TODO: additional unit tests? test changing objective, setting 
#       solver options, maximization, changing constraints, copying,
#       Hessian approximation


