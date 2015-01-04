"""Test function objects and their differentiation. 

These are a bit tricky as differentiation to a string is
underdetermined: 'x*2', '2*x', 'x*2.0', '2.0*x', etc., are all
perfectly good derivatives of 'x**2' with respect to 'x'. Thus in some
places here we are simply checking that these differentiation methods
return what you would get by differentiating the expression
appropriately with expr_manip-- not that those derivatives are in fact
correct.

"""

from nose.tools import assert_almost_equals
from fluxtools.functions import Function, Linear
import fluxtools.expr_manip as em
em_zero_string = em.diff_expr('x','y')

def test_cache_persistence():
    f1 = Function('x**2 * y**2')
    all_first_derivs = f1.all_first_derivatives()
    all_second_derivs = f1.all_second_derivatives()
    f2 = Function('a+b')
    assert f2._first_derivatives == {}
    assert f2._second_derivatives == {}
    assert f2.derivative('x') == em_zero_string

class Linear_Test():
    def setup(self):
        self.f = Linear(dict(zip('abcde',range(5))))
    def test_coefficient_attribute(self):
        assert self.f.coefficients == dict(zip('abcde',
                                               map(str, range(5))))
    def test_variable_extraction(self):
        assert self.f.variables == set('abcde')
    def test_some_first_derivatives(self):
        assert self.f.all_first_derivatives('defg') == {'d': '3',
                                                        'e': '4'}
    def test_second_derivatives(self):
        assert self.f.all_second_derivatives() == {}
    def test_substitution(self):
        d = {'e': 'eta', 'f': 'phi'}
        g = self.f.substitute(d)
        x0 = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'eta': 1}
        assert_almost_equals(eval(g.code(), x0), 4.)
        assert g._first_derivatives == {'a': '0', 'b': '1', 'c': '2', 
                                        'd': '3', 'eta': '4'}

# TODO: test long linear combinations.
    
class Biquadratic_Test():
    def setup(self):
        self.expr = 'a**2*x**2'
        self.vars = set(('x','a'))
        self.f = Function(self.expr,name='biquadratic')
        self.x0 = {'a': 2., 'x': 4.}
    def test_variable_extraction(self):
        assert self.f.variables == self.vars
    def test_single_first_derivative(self):
        assert self.f.derivative('x') == em.diff_expr(self.expr, 'x')
    def test_zero_first_derivative(self):
        assert self.f.derivative('y') == em_zero_string
    def test_all_first_derivatives(self):
        derivs = self.f.all_first_derivatives()
        assert derivs == {v:em.diff_expr(self.expr,v) for v in self.vars}
    def test_some_first_derivatives(self):
        derivs = self.f.all_first_derivatives(('x','y'))
        assert derivs.keys() == ['x']
    def test_single_second_derivative(self):
        exact = em.diff_expr(em.diff_expr(self.expr, 'a'),'x')
        assert self.f.second_derivative(('a','x')) == exact
    def test_zero_second_derivative(self):
        assert self.f.second_derivative(('b','x')) == em_zero_string
    def test_all_second_derivatives(self):
        keys = [('x','x'),('a','x'),('a','a')]
        exact = {t: em.diff_expr(em.diff_expr(self.expr, t[0]),
                                 t[1]) for t in keys}
        assert self.f.all_second_derivatives() == exact
    def test_some_second_derivatives(self):
        derivs = self.f.all_second_derivatives(('x','y'))
        assert derivs.keys() == [('x','x')]
    def test_first_deriv_cache(self):
        assert 'x' not in self.f._first_derivatives
        self.f.derivative('x')
        assert 'x' in self.f._first_derivatives
        self.f._first_derivatives['x'] = '20'
        assert self.f.derivative('x') == '20'
        assert self.f.all_first_derivatives()['x'] == '20'
    def test_first_deriv_cache(self):
        assert ('a','x') not in self.f._second_derivatives
        self.f.second_derivative(('a','x'))
        assert ('a','x') in self.f._second_derivatives
        self.f._second_derivatives[('a','x')] = '20'
        assert self.f.second_derivative(('a','x')) == '20'
        assert self.f.all_second_derivatives()[('a','x')] == '20'
    def test_code(self):
        assert_almost_equals(eval(self.f.code(),self.x0), 64.)
    def test_derivative_code(self):
        assert_almost_equals(eval(self.f.first_derivative_code('x'),
                                  self.x0), 32.)
    def test_second_derivative_code(self):
        assert_almost_equals(eval(self.f.second_derivative_code(('x','x')),
                                   self.x0), 8.)
    def test_code_cache(self):
        assert not self.f._code
        self.f.code()
        assert self.f._code
        self.f._code = compile('1.','<string>','eval')
        assert_almost_equals(eval(self.f.code(),self.x0),1.)
    def test_derivative_code_cache(self):
        assert not 'x' in self.f._first_derivative_code
        self.f.first_derivative_code('x')
        assert 'x' in self.f._first_derivative_code
        self.f._first_derivative_code['x'] = compile('1.','<string>','eval')
        assert_almost_equals(eval(self.f.first_derivative_code('x'),self.x0),1.)
    def test_second_derivative_code_cache(self):
        assert ('x','x') not in self.f._second_derivative_code
        self.f.second_derivative_code(('x','x'))
        assert ('x','x') in self.f._second_derivative_code
        self.f._second_derivative_code[('x','x')] = compile('1.',
                                                            '<string>','eval')
        assert_almost_equals(eval(self.f.second_derivative_code(('x','x')),
                                   self.x0), 1.)

    def test_substitution(self):
        self.f.all_first_derivatives()
        self.f.all_second_derivatives()
        d = {'x': 'gamma', 'y': 'beta'}
        g1 = self.f.substitute(d)
        g2 = Function('a**2*gamma**2')
        g2.all_first_derivatives()
        g2.all_second_derivatives()
        assert g1.math == g2.math
        assert g1.variables == g2.variables
        assert g1._first_derivatives == g2._first_derivatives
        assert g1._second_derivatives == g2._second_derivatives
        assert g1.name == 'biquadratic'
