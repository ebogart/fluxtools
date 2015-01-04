"""Generate nonlinear programs, typically from reaction networks; solve them.

Exported classes:
-----------------
NonlinearModel : nonlinear optimization problem instance
NonlinearNetworkModel : network-specific NonlinearModel
OptimizationFailure: raised when a call to an optimization routine
    leads to an unsatisfactory result

Exported functions:
-------------------
tree_sum : write a very defensively parenthesized sum of terms to avoid
    hitting the recursion depth limit when parsing

Module-level attribute:
-----------------------
default_ipopt_options : dict of options used by default by NonlinearModel 
   instances, overridden by options specified in the ipopt_options
   attribute of a particular instance, the options argument of solve(),
   or the ipopt.opt file in the working directory.
   
"""
######################################################################
# HISTORICAL NOTES
#
# This branch revised as of 4.22.2012 to regress to python 2.6
# compatibility by removing dictionary comprehensions. (Note they were
# later added back in again and the module is no longer
# 2.6-compatible.)
#
# 7.6.2012: renovating the method of dynamic Python function
# construction to reduce the memory required in the compilation
# process.
#
# October 2012: second derivative caching to reduce calculation time
# in creating replica models introduced
#
# November 2012: Change core optimization call to reflect the fact
# that solve() now returns constraint Lagrange multipliers.
#
# March 2013: Add PerturbationModel class
#
# Early 2014: disentanglement from SloppyCell, other refactoring and
# cleanup, better-designed Function class which recognizes the
# fundamental interchangeability of objective and constraints.

# To do: integrate linear modeling in same framework; detailed
# information about the nature of the failure in the
# OptimizationFailure class.

######################################################################
# IMPORTS

import math
import re
import copy
import warnings

import numpy as np
import pyipopt 

import expr_manip as em
from reaction_networks import * 
from keyedlist import KeyedList
from functions import Function, Linear

########################################################################
# CONSTANTS

# Set effective infinite values per IPOPT documentation
IPOPT_INF = 2.0e19
IPOPT_MINUS_INF = -2.0e19

# Specify default options for nonlinear programs.
# This is intended to be modified after import, to 
# choose (eg) a solver for all problems within a program.
default_ipopt_options = {}

######################################################################
# EXCEPTIONS
class OptimizationFailure(Exception):
    pass


######################################################################
# MATH HANDLING

# Define some useful functions (borrowing wholesale from Network_mod.py)
_common_namespace = {'log': np.log,
                     'log10': np.log10,
                     'exp': math.exp,
                     'cos': math.cos,
                     'sin': math.sin,
                     'tan': math.tan,
                     'acos': math.acos,
                     'asin': math.asin,
                     'atan': math.atan,
                     'cosh': math.cosh,
                     'sinh': math.sinh,
                     'tanh': math.tanh,
                     # These don't have C support, so I've dropped them for
                     # now.                                                 
                     #'arccosh': scipy.arccosh,                             
                     #'arcsinh': scipy.arcsinh,                             
                     #'arctanh': scipy.arctanh,                             
                     'pow': math.pow,
                     'sqrt': math.sqrt,
                     'exponentiale': math.e,
                     'pi': math.pi,
                     #'scipy': scipy, # why was this necessary originally?
                     #'operator': operator,
                     'min': np.min,
                     'max': np.max
                     }
_standard_func_defs = [('root', ['n', 'x'],  'x**(1./n)'),
                       ('cot', ['x'],  '1./tan(x)'),
                       ('arccot', ['x'],  'atan(1/x)'),
                       ('coth', ['x'],  '1./tanh(x)'),
                       #('arccoth', ['x'],  'arctanh(1./x)'),               
                       ('csc', ['x'],  '1./sin(x)'),
                       ('arccsc', ['x'],  'asin(1./x)'),
                       ('csch', ['x'],  '1./sinh(x)'),
                       #('arccsch', ['x'],  'arcsinh(1./x)'),               
                       ('sec', ['x'],  '1./cos(x)'),
                       ('arcsec', ['x'],  'acos(1./x)'),
                       ('sech', ['x'],  '1./cosh(x)'),
                       #('arcsech', ['x'],  'arccosh(1./x)'),               
                       ]
_common_func_strs = {} 
for id, vars, math in _standard_func_defs:
    var_str = ','.join(vars)
    func = 'lambda %s: %s' % (var_str, math)
    _common_func_strs[id] = func
        # These are all the partial derivatives                                 
    # Do I ever use the following?
    for ii, wrt in enumerate(vars):
        deriv_id = '%s_%i' % (id, ii)
        diff_math = em.diff_expr(math, wrt)
        func = 'lambda %s: %s' % (var_str, diff_math)
        _common_func_strs[deriv_id] = func

# ... skip Network_mod C support
for func_id, func_str in _common_func_strs.items():
    _common_namespace[func_id] = eval(func_str, _common_namespace, {})

# Load all the new function definitions into the module's global namespace
globals().update(_common_namespace)

######################################################################
# BASE NONLINEAR PROGRAM CLASS

class NonlinearModel:

    """ A general nonlinear optimization problem object.
    
    Currently, this generates an IPOPT problem instance through pyipopt.
    Eventually it may be desirable to separate the solver interface
    from the python representation of the problem, to allow different 
    solvers, or different interfaces to IPOPT to be used. 

    Variables and parameters effectively share a namespace with 
    constraint functions (expand on this later.)

    """

    def __init__(self):

        """Initialize an empty model. """

        self.constraints = KeyedList()
        self.objective_function = None

        # List bounds on variables and constraint functions.
        # Where upper (lower) bound is not specified, or is None, it is 
        # treated as positive (negative) infinity. 
        self.upper_bounds = {} 
        self.lower_bounds = {} 
        
        # Parameters are effectively variables with a fixed value;
        # they may be referred to in functions but are not optimized 
        # over.
        self.parameters = {}

        # The self.active_nlp flag tracks whether a non-closed pyipopt
        # problem instance is attached as self.nlp. This is important because
        # self.nlp.close() needs to be called to free up the (often 
        # substantial) memory used by the problem instance before self.nlp
        # is set to a new problem instance (the problems can't be properly
        # garbage collected), but attempting to close an already closed
        # problem leads to an immediate segmentation fault.
        self.active_nlp = False  

        # Track whether the problem is in a compiled state (lists of
        # code objects and constant terms for functions/derivatives,
        # variable and constraint counters, etc., are present and
        # should be copied, if so.)
        self._compiled = False

        # Choose which IPOPT option for the handling of variables with
        # fixed values we should try to require. Note that this may be
        # overridden if the structure of the problem requires it.
        # 'make_parameter' is IPOPT's default; I add it here as an
        # attribute so that it may be set to other values at need. 
        self.fixed_variable_treatment = 'make_parameter'

        # Allow IPOPT options in general to be set persistently
        # for problems associated with this model in an attribute.
        # Note at this level we do not distinguish between string,
        # integer, and floating-point options; the type of each option
        # value will be checked before solving, and the appropriate 
        # method of the pyipopt nlp instance will be called.
        self.ipopt_options = {}

    def resolve_name(self, entity_name):
        """Interpret an interactively supplied entity name.

        This may be a variable or parameter name, or the name of 
        a constraint function; not, in general, the name of the objective
        function, if that is even defined.
        
        This exists to be overridden by subclasses, allowing friendlier
        interactive references to often cumbersome internal variable names.

        """
        return entity_name

    def get_upper_bound(self, v):
        """ Get the upper bound on the variable/constraint v."""
        return self.upper_bounds.get(self.resolve_name(v),None)

    def get_lower_bound(self,v):
        """Get the lower bound on the variable/constraint v."""
        return self.lower_bounds.get(self.resolve_name(v),None)

    def get_bounds(self,v):
        """ Get a tuple containg bounds on variable/constraint v.

        Returns (lower_bound, upper_bound).
        
        """
        return (self.get_lower_bound(v),
                self.get_upper_bound(v))

    def set_upper_bound(self, v, bound):
        """Set the upper bound on the variable/constraint v."""
        self.upper_bounds[self.resolve_name(v)] = bound 

    def set_lower_bound(self, v, bound):
        """Set the lower bound on the variable/constraint v."""
        self.lower_bounds[self.resolve_name(v)] = bound

    def set_equality(self, v, bound):
        """ Set equal lower and upper bounds on variable/constraint v. """
        self.set_upper_bound(v, bound)
        self.set_lower_bound(v, bound)

    def set_inequality(self, v, lower_bound, upper_bound):
        """ Set lower and upper bounds on the variable or constant v. """
        self.set_upper_bound(v, upper_bound)
        self.set_lower_bound(v, lower_bound)

    def set_bound(self, v, bound):
        """ Set bounds on variable/constraint v with flexible syntax.

        If the bound is a scalar or None, it is interpreted as an equality;
        if a tuple, as (lower, upper) inequality bounds.

        """
        if isinstance(bound, tuple):
            self.set_inequality(v, bound[0], bound[1])
        else:
            self.set_equality(v, bound)

    def set_bounds(self, bounds):
        """ Set bounds on many variables/constraints from a dict. """
        for v, bound in bounds.iteritems():
            self.set_bound(v,bound)

    def set_objective(self, objective_id, expression):
        """ Set the objective function of the problem. 

        The existing objective function will be lost.

        """

        self.objective_function = Function(expression,
                                           name=objective_id)

    def add_constraint(self, function_id, expression, value=None):
        """ Add the expression to the problem as a constraint function. 

        Optionally, set the value to which the function is constrained,
        (a scalar, or (lower bound, upper bound) tuple of scalars or Nones.)

        """
        self.constraints.set(function_id, 
                             Function(expression, name=function_id))
        if value is not None:
            if isinstance(value, tuple):
                self.set_inequality(function_id, *value)
            else:
                self.set_equality(function_id, value)

    def remove_constraint(self, function_id):
        """ Delete the indicated constraint from the problem. 

        Any bounds on this constraint will be cleared.

        """
        self.constraints.remove_by_key(function_id)
        self.upper_bounds.pop(function_id, None)
        self.lower_bounds.pop(function_id, None)

        
    def write_bounds_to_file(self, filename):
        """Write current variable and function bounds to a text file.

        Each line will contain the entity internal name, its lower bound or 
        'None', and its upper bound or 'None', separated by tabs.

        All entries in self.upper_bounds or self.lower_bounds will 
        be written to the file, unless both bounds are None, or the key
        is not currently a variable or function id. 
        
        """
        bound_keys = set(self.lower_bounds.keys())
        bound_keys.update(set(self.upper_bounds.keys()))
        
        lines = []
        for k in bound_keys:
            if k in self.variables or k in self.functions.keys(): 
                l = self.lower_bounds(k)
                u = self.upper_bounds(k)
                if not (l is None and u is None):
                    lines.append('\t'.join([k, repr(l), repr(u)]) + '\n')
        
        with open(filename,'w') as f:
            f.write('\n'.join(lines) + '\n')

    def make_variable_bound_vectors(self):

        """ Prepare vectors of upper and lower bounds on variables. 
        
        Returns: 
        x_L, x_U -- numpy arrays of upper and lower bounds on self.variables

        """

        x_L = []
        x_U = []

        for v in self.variables:
            # A variable which is unbounded may be either not included 
            # in the dictionary of bounds or have a bound of None. In 
            # either case, we need to give it an 'infinite' bound 
            # using the effective infinities defined above.

            lower_bound = self.get_lower_bound(v)
            if lower_bound is None:
                lower_bound = IPOPT_MINUS_INF
            x_L.append(lower_bound)

            upper_bound = self.get_upper_bound(v)
            if upper_bound is None:
                upper_bound = IPOPT_INF
            x_U.append(upper_bound)

        return np.array(x_L), np.array(x_U) 

    def make_constraint_bound_vectors(self):
        """ Prepare vectors of upper and lower bounds on constraints. 
        
        Returns: 
        g_L, g_U -- numpy arrays of upper and lower bounds on self.constraints

        """
        g_L = []
        g_U = []
        for constraint in self.constraints.keys():
            # As with bounds on variables, unbounded constraint functions
            # may either have explicit bounds of None or be excluded
            # from the dictionaries of bounds entirely.

            lower_bound = self.lower_bounds.get(constraint, None)
            if lower_bound is None:
                lower_bound = IPOPT_MINUS_INF
            g_L.append(lower_bound)

            upper_bound = self.upper_bounds.get(constraint, None)
            if upper_bound is None:
                upper_bound = IPOPT_INF
            g_U.append(upper_bound)

        g_L = np.array(g_L) 
        g_U = np.array(g_U) 

        return g_L, g_U

    def correct_guess(self, x0):
        """Project a vector of variable values inside the bounds.

        This may be useful for example in setting up an initial guess from
        which to start the optimizer. Note that IPOPT should automatically 
        project all variables inside their bounds at initialization, but
        this has proven helpful nonetheless, eg in situations where
        some bounds have been relaxed by IPOPT (due to fixed_variable_treatment
        relax_bounds.) 

        This function considers only consistency with bounds
        on variables, not satisfaction of equality/inequality constraints.

        Arguments: 
        x0 -- list or array of length of self.variables
        Returns: 
        x1 -- modified copy of x0.

        """
        x1 = x0.copy()
        for i,x in enumerate(x0):
            l,u = self.get_bounds(self.variables[i])
            if l is not None and x < l:
                x1[i] = l
            if u is not None and x > u:
                x1[i] = u

        return x1

    #######
    # CORE IPOPT INTERFACE CODE

    def _update_namespace(self, x):
        """ Fill self._namespace with the values of variables, parameters at x. """
        self._namespace = dict(zip(self.variables, x))
        self._namespace.update(self.parameters)

    def _eval(self, code):
        """ Evaluate code object (or expression) at the current point.
        
        The code is evaluated in the context of nlcm._common_namespace and 
        self._namespace.
        
        """
        return eval(code, _common_namespace, self._namespace)

    def update_variables(self):
        """Find the decision variables of the problem and order them.

        All the functions in the problem (constraints and objective)
        are searched for the variables on which they depend, and those
        which are in self.parameters are excluded.

        The variables are then placed, sorted, into the tuple
        self.variables, and the dictionary self.var_index is set up to
        map a variable string to its index in the tuple. The variable
        counter s.nvar is updated.

        """

        variables = set() 

        for f in self.constraints:
            variables.update(f.variables)
            # faster, if less reliable, than
            # variables.update(em.extract_vars(f.math))
        variables.update(self.objective_function.variables)

        variables = variables - set(self.parameters)

        # Use a tuple to make clear that this ordering should not be
        # modified casually.
        variables = tuple(sorted(variables))
        self.variables = variables
        self.nvar = len(variables)

        var_index = {}
        for i, v in enumerate(variables): 
            var_index[v] = i
        self.var_index = var_index

    def compile(self):
        """Compile objective, constraint, and derivative functions.
       
        Lists of compiled code objects and constants representing
        variable and constant terms in the objective and constraint
        functions and their first and second derivatives, indices of
        nonzero terms, etc., are prepared for use by the methods
        self.eval_f, self.eval_g, etc. Miscellaneous attributes
        (e.g. the counters of variables, constraints, and nonzero Jacobian
        and Hessian elements, self.nvar, self.ncon, self._nnzj, self._nnzh)
        are set.

        """
        self.update_variables() # also sets self.nvar
        
        # Set up the objective and its gradient
        self.setup_f() 
        self.setup_grad_f()
        # Set up constraints and derivatives 
        self.setup_g() 
        self.ncon = len(self.constraints)
        self.setup_jac_g() # also sets self._nnzj
        # Set up the Hessian.
        self.setup_h() # also sets self._nnzh

        self._compiled = True
        
    def setup_f(self):
        """ Set up the objective function, ensuring its code is compiled. """
        # Calling the code() method compiles the function and caches the result,
        # if this has not been done already.
        self.objective_function.code()

    def eval_f(self, x):
        """ Evaluate the objective function at point x. """
        self._update_namespace(x)
        return self._eval(self.objective_function.code())
        
    def setup_grad_f(self):
        """Find constant and variable terms of the gradient of the objective.
        
        This method populates the (self.nvar,) array
        self._grad_f_constant_terms with the values of any constant
        terms of the gradient (it is otherwise zero) and the list
        self._grad_f_variable_terms with tuples (index,
        code_object_for_that_index_in_grad_f).

        """
        constant_terms = np.zeros(self.nvar)
        variable_terms = []
        all_variables = set(self.variables)
        derivatives = self.objective_function.all_first_derivatives(all_variables)
        for variable, derivative in derivatives.iteritems():
            variable_index = self.var_index[variable]
            if em.extract_vars(derivative):
                code = self.objective_function.first_derivative_code(variable)
                variable_terms.append((variable_index, code))
            else:
                constant_terms[variable_index] = np.float(derivative)
        self._grad_f_constant_terms = constant_terms
        self._grad_f_variable_terms = variable_terms
        
    def eval_grad_f(self, x):
        """ Evaluate the gradient of the objective function.

        Returns an array of shape (self.nvar,) containing the gradient
        at x. 

        """
        self._update_namespace(x)
        gradient = self._grad_f_constant_terms.copy()
        for i, code in self._grad_f_variable_terms:
            gradient[i] += self._eval(code)
        return gradient

        
    def setup_g(self):
        """ Set up constraint functions by ensuring their code is compiled. """
        for constraint_function in self.constraints:
            constraint_function.code()

    def eval_g(self, x):
        self._update_namespace(x)
        return np.array([self._eval(constraint.code()) for 
                         constraint in self.constraints], dtype=np.float)
        
    def setup_jac_g(self):
        """List nonzero derivatives of the constraints, corresponding code objects. 

        We determine the (constraint_index, variable_index) pairs
        corresponding to (potentially) nonzero constraint derivatives
        and list them in self._jac_g_indices, setting self._nnzj to
        the length of this list. Then, the array of length nnzj
        self._jac_g_constant_terms is initialized with the numerical
        values of the terms which are constant (elsewhere zero), and
        self._jac_g_variable_terms is set to a list of (index,
        code_object) tuples, where the indices are the positions of
        the variable terms in the overall list of nozero Jacobian
        terms. 

        """
        indices = []
        constant_terms = []
        variable_terms = []
        all_variables = set(self.variables)        

        term = 0 # index of the current nonzero term
        for i, constraint in enumerate(self.constraints):
            derivatives = constraint.all_first_derivatives(all_variables)
            for variable, expression in derivatives.iteritems():
                j = self.var_index[variable]
                indices.append((i,j))
                if em.extract_vars(expression):
                    code = constraint.first_derivative_code(variable)
                    variable_terms.append((term, code))
                    constant_terms.append(0.)
                else:
                    constant_terms.append(expression_to_float(expression))
                term += 1
            
        self._nnzj = len(indices)
        self._jac_g_indices = indices
        self._jac_g_constant_terms = np.array(constant_terms, dtype=np.float)
        self._jac_g_variable_terms = variable_terms

    def eval_jac_g(self, x, flag=False):
        """ Evaluate or list nonzero elements of the constraint Jacobian. 

        Arguments:
        x - numpy array of length self.nvar
        flag - Boolean indicating whether the indices of the nonzero elements
             should be returned.

        Returns: if flag is not set, a numpy array of the values of the 
        (possibly) nonzero elements of the Jacobian at x. If flag is set,
        a 2-tuple of numpy arrays, the first giving the constraints 
        and the second the variables associated with the nonzero Jacobian 
        elements, is returned.

        """
        if flag:
            return tuple(map(np.array,zip(*self._jac_g_indices)))

        self._update_namespace(x)
        gradient = self._jac_g_constant_terms.copy()
        for k, code in self._jac_g_variable_terms:
            gradient[k] += self._eval(code)
        return gradient

    def setup_h(self):
        """Identify nonzero second derivatives of objective and constraints. 
        
        Pairs of variables with respect to which the second derivatives of
        any such function are not (necessarily) zero are identified and a
        corresponding list of pairs of indices (i,j), i>j, into the upper
        triangular part of the Hessian is stored as self._h_indices. 
        
        self._nnzh is set to the length of this list.
        
        The lists self._h_objective_constant_terms and
        self._h_objective_variable_terms are populated with tuples
        (relevant_index_into_self.h_indices, term) where the term is
        either a numpy float or a compiled code object as appropriate.

        The lists self._h_constraint_variable_terms and
        self._h_constraint_constant_terms are populated with tuples
            (relevant_index_into_self.h_indices, 
             constraint_index,
             term) 
        where the term is a float or code object as appropriate.

        """
        all_variables = set(self.variables)
        objective_terms = self.objective_function.all_second_derivatives(all_variables)
        constraint_terms = {i: constraint.all_second_derivatives(all_variables)
                            for i, constraint in enumerate(self.constraints)}
        keys = set(objective_terms)
        for i, derivatives in constraint_terms.iteritems():
            keys.update(derivatives)

        self._nnzh = len(keys)

        key_to_index_pair = {(v1,v2): tuple(sorted((self.var_index[v1],
                                               self.var_index[v2])))
                             for v1, v2 in keys}
        self._h_indices = sorted(key_to_index_pair.values())
        index_pair_to_h_index = dict([(pair, i) for i, pair in
                                      enumerate(self._h_indices)])
        key_to_h_index = {k: index_pair_to_h_index[key_to_index_pair[k]]
                          for k in keys}

        objective_constant_terms = []
        objective_variable_terms = []
        for k, expression in objective_terms.iteritems():
            if em.extract_vars(expression):
                code = self.objective_function.second_derivative_code(k)
                objective_variable_terms.append((key_to_h_index[k], code))
            else:
                objective_constant_terms.append((key_to_h_index[k], 
                                                 expression_to_float(expression)))

        self._h_objective_constant_terms = objective_constant_terms
        self._h_objective_variable_terms = objective_variable_terms

        constraint_constant_terms = []
        constraint_variable_terms = []
        for i, terms in constraint_terms.iteritems():
            for k, expression in terms.iteritems():
                if em.extract_vars(expression):
                    code = self.constraints[i].second_derivative_code(k)
                    constraint_variable_terms.append((key_to_h_index[k],
                                                      i,
                                                      code))
                else:
                    constraint_constant_terms.append(
                        (key_to_h_index[k], i,
                         expression_to_float(expression))
                    )
        
        self._h_constraint_constant_terms = constraint_constant_terms
        self._h_constraint_variable_terms = constraint_variable_terms
        
    def eval_h(self, x, lagrange, obj_factor, flag=None):

        """Evaluate or list nonzero elements of the Hessian.

        Specifically, what is evaluated is the Hessian of the Lagrangian 
        in the particular, slightly unconventional, form described in
        eqn. 9 of the section "Interfacing with IPOPT through code" of the 
        IPOPT online documentation. 

        x - numpy array of length self.nvar
        lagrange - array of length self.ncon giving Lagrange multipliers 
            for the constraints
        obj_factor - float, used to scale the contributions of the
            objective function to the Hessian
        flag - Boolean indicating whether the indices of the nonzero elements
             should be returned.

        Returns: if flag is not set, a numpy array of the values of the 
        (possibly) nonzero elements of the Hessian at x, appropriately
        scaled by the Lagrange multipliers and objective function factor.

        If flag is set, a 2-tuple of numpy arrays, giving first
        and second indices respectively for the possibly nonzero
        entries of the upper triangular part of the Hessian matrix, is returned.

        """
        # This could be sped up in various ways, most notably if the
        # objective factor or Lagrange multipliers are sometimes
        # exactly zero, or close enough that we are comfortable
        # treating them as zero.
        if flag: 
            return tuple(map(np.array, zip(*self._h_indices)))

        self._update_namespace(x)
        h = np.zeros(self._nnzh)
        
        # First populate with the terms related to the objective,
        # unscaled. 
        if self._h_objective_constant_terms:
            indices, values = zip(*self._h_objective_constant_terms)
            # indices is a tuple, but numpy array indexing treats
            # tuples differently from lists; we want an index 
            # array rather than a multi-dimensional index, so 
            # we cast to list explicitly.
            h[list(indices)] = values
        for index, code in self._h_objective_variable_terms:
            h[index] = self._eval(code)

        # Then scale by the objective factor
        h *= obj_factor

        # Then add the terms associated with the constraints, scaling 
        # by the Lagrange multipliers
        for index, constraint_index, value in self._h_constraint_constant_terms:
            h[index] += lagrange[constraint_index] * value
        for index, constraint_index, code in self._h_constraint_variable_terms:
            h[index] += lagrange[constraint_index] * self._eval(code)
        return h 

    def repeated_solve(self, x0, max_iter, max_attempts, options={}):
        """ Solve the nonlinear problem, restarting as needed.

        In some cases it is advantageous to stop the progress of the solver
        and restart it from the same point, resetting the internal variable
        scaling. (There are probably better ways to do this, but this works
        in practice.) Here, the solver will be restarted if it exits with 
        status -1 (iteration limit exceeded) or -2 (restoration failure).

        Arguments:
        x0 -- starting point
        max_iter -- how many iterations to run before restarting (passed 
           to IPOPT as max_iter)
        max_attempts -- how many times to restart before giving up
        options -- additional options for the solve() method.

        """

        all_options = {'max_iter': max_iter}
        all_options.update(options)
        attempt = 0
        x = x0
        while attempt < max_attempts:
            try:
                x = self.solve(x0=x, options=all_options)
            except OptimizationFailure as e:
                if self.status not in {-1, -2}:
                    raise e
                else:
                    x = self.x.copy()
            else:
                return x
            attempt += 1
        # We have made the maximum number of restart attempts
        # but still not converged, or still ended up with a restoration 
        # failure.
        raise OptimizationFailure('No convergence after %d attempts (status %d)' %
                                  (max_attempts, self.status))

    def solve(self, x0=None, options={}):
        """Solve the nonlinear problem from a specified starting point.

        Arguments:
        x0 -- array of length self.nvars with initial guesses for values of
        self.variables (in that order), or a scalar to be used as the initial
        guess for all variables; if not supplied, all variables will be 
        set initially to 1.0.
        options -- dictionary {'option': value} of options to pass to 
        IPOPT. Options will be taken from (in increasing order of priority)
        IPOPT's default values, the module-wide default_ipopt_options, those
        specified in self.fixed_variable_treatment or self.ipopt_options, 
        this argument, or the ipopt.opt file (if any) in the working directory.
        Errors will result if invalid names or values for options are given 
        here.
        
        Returns: 
        x -- array of variable values returned by IPOPT (ordered as in 
        self.variables)

        Also sets self.x, self.zl, self.zu, self.constraint_multipliers, 
        self.obj_value, and self.status to x, the lower and upper
        bound multipliers, the Lagrange multipliers associated with
        the constraint functions, the final objective function value,
        and the optimzation status, respectively, and sets self.soln
        to a dictionary mapping each variable key to its value in
        self.x.
   
        This method does not recompile the objective, constraint and
        derivative functions if self.compile() has already been
        called. If anything except bounds on variables, bounds on
        constraints, and/or parameter values has been changed since
        the last time self.compile() has been called, self.compile
        must be called again before solving, or this method will
        attempt to solve an out-of-date version of the problem.

        Each call to solve does create a new pyipopt problem instance
        as self.nlp, closing the old one beforehand if self.active_nlp 
        is true, and resetting self.active_nlp to true afterwards.

        """

        if not self._compiled:
            self.compile()

        if x0 is None:
            x0 = np.ones(self.nvar)
        elif np.isscalar(x0):
            x0 = x0 * np.ones(self.nvar)
        else:
            # Assume this is a vector. Check its length, as trying to
            # proceed with an inappropriately sized starting point may
            # lead to crashes.
            if len(x0) != self.nvar:
                message = ('Starting point has wrong dimension (needed %d, given %d)' % 
                           (self.nvar, len(x0)))
                raise ValueError(message)
        
        self.close_nlp()

        x_L, x_U = self.make_variable_bound_vectors()
        g_L, g_U = self.make_constraint_bound_vectors()
        
        # The following avoids a segmentation fault that results
        # when bound methods are supplied as evaluation functions 
        # to pyipopt, which I don't really understand:
        eval_f = lambda x, user_data=None: self.eval_f(x)
        eval_grad_f = lambda x, user_data=None: self.eval_grad_f(x)
        eval_g = lambda x, user_data=None: self.eval_g(x)
        eval_jac_g = lambda x, flag, user_data=None: self.eval_jac_g(x,flag)
        eval_h= lambda x, lagrange, obj_factor, flag, user_data=None: \
            self.eval_h(x,lagrange,obj_factor,flag)

        self.nlp = pyipopt.create(self.nvar, x_L, x_U, 
                                  self.ncon, g_L, g_U,
                                  self._nnzj, self._nnzh, 
                                  eval_f, eval_grad_f, eval_g,
                                  eval_jac_g, eval_h)

        self.active_nlp = True
        # Handle generic IPOPT options
        all_options = {}
        all_options.update(default_ipopt_options)
        all_options['fixed_variable_treatment'] = self.fixed_variable_treatment
        all_options.update(self.ipopt_options)
        all_options.update(options)
        for option, value in all_options.iteritems():
            if isinstance(value, str):
                self.nlp.str_option(option, value)
            elif isinstance(value, int):
                self.nlp.int_option(option, value)
            else:
                self.nlp.num_option(option, value)

        (self.x, self.zl, self.zu, self.constraint_multipliers, 
         self.obj_value, self.status) = self.nlp.solve(x0)

        self.soln = dict(zip(self.variables, self.x))
        
        if self.status not in (0, 1):
            raise OptimizationFailure('IPOPT exited with status %d' % self.status)

        return self.x.copy()

    def close_nlp(self):
        """Close the nonlinear programming instance if one appears to be open.

        If self.active_nlp is True, call sell self.nlp.close() and reset 
        self.active_nlp to False; otherwise pass. Possibly a warning ought
        to be issued, as if the nlp instance really is open somewhere, 
        failure to close it will leak memory, but at the moment this 
        is omitted to avoid spurious warnings when setting the object's
        first nlp instance.

        """

        if self.active_nlp:
            self.nlp.close()
            self.active_nlp = False
    # END CORE IPOPT INTERFACE CODE
    ######

    def __getstate__(self):
        """ Return a dict of attributes for copying, excluding the NLP. """
        state = self.__dict__.copy()
        if 'nlp' in state:
            del state['nlp']
            state['active_nlp'] = False
        return state

    def copy(self):
        """ Return a (deep) copy of the NLP instance. """
        return copy.deepcopy(self)

######################################################################
# WORKING WITH NETWORKS
def get_components(network):
    """Return a list of components of the network we care about.

    Currently lists the network's species, parameters,
    compartments,reactions, function definitions, events and
    constraints. (Missing: rules.)

    """
    components = (network.species + network.parameters +
                  network.compartments + network.reactions +
                  network.functionDefinitions + network.events +
                  network.constraints)

    return list(components)

class NonlinearNetworkModel(NonlinearModel):

    """Nonlinear constraint model based on a network.

    Extends NonlinearModel to allow construction of a model from a
    reaction_networks Network object, tracking of variables that
    correspond to species and reactions, reaction stoichiometries and
    conservation rules, etc.
    
    TODO: handle sbml Parameter entities in the network appropriately.

    """

    def __init__(self, model_id, net, external_species_ids = []):

        """Initialize the model from reaction_networks.Network.

        Effectively, this imposes constraints based on conservation of
        species in the network (other than those given in
        external_species_ids), saves copies of the network's species
        and reaction lists for local reference, and guarantees that
        all variables in the model which are species in the network
        are taken as having a lower bound of 0.0 unless explicitly set
        to None or some other value.

        Currently the network is not maintained as the model changes
        and no provision for SBML reexport is made. 
        
        """

        NonlinearModel.__init__(self) 
        self.id = model_id

        self.species = net.species.copy()
        self.species_set = set(self.species.keys())
        self.reactions = net.reactions.copy()

        self.add_conservation_functions()
        self.conserve(*(self.species_set 
                        - set(external_species_ids)))

    def get_lower_bound(self, v): 
        """ Get the lower bound on variable/constraint v.

        Ensures variables which are species in the network are bounded below
        by 0.0 unless another value has been explicitly specified.
        
        """
        internal_v = self.resolve_name(v)
        if internal_v in self.lower_bounds:
            return self.lower_bounds[internal_v]
        elif internal_v in self.species_set:
            return 0.
        else:
            return None
            
    def conservation_function_id(self, species):
        """ Find the id of the constraint enforcing conservation of a species.

        """
        return '_conservation_' + self.resolve_name(species)

    def conserve(self, *species_list):
        """ Require that species be at steady state.
        
        For each species provided as an argument, the corresponding
        conservation constraint function created by add_conservation_functions
        will be constrained to equal zero.

        If no species are listed, every species in self.net will be 
        conserved.

        """
        if not species_list:
            species_list = self.net.species.keys()
        
        for s in species_list:
            conservation_function_id = self.conservation_function_id(s)
            self.upper_bounds[conservation_function_id] = 0.0
            self.lower_bounds[conservation_function_id] = 0.0

    def do_not_conserve(self, *species_list):
        """Relax requirements that species be at steady state.
        
        For each species provided as an argument the corresponding
        conservation constraint function created by add_conservation_functions
        will be made unbounded (overriding the previous upper and lower
        bounds, whether those were zero or not!)

        If no species are listed, remove conservation boundaries from
        all species in self.net. 
        
        """
        if not species_list:
            species_list = self.net.species.keys()
        
        for s in species_list:
            conservation_function_id = self.conservation_function_id(s)
            self.upper_bounds.pop(conservation_function_id,None)
            self.lower_bounds.pop(conservation_function_id,None)

    def add_conservation_functions(self, *species_list):
        """ Add functions to the network representing net species production.

        The functions give the sum of fluxes to each species.

        Functions are of the form '_conservation_' + species_id.

        Species should be specified by id, to ensure that the resulting
        function name is a valid python/C variable id.
        
        If the species list is not specified, conservation functions will
        be created for all species in the associated network.

        """

        if not species_list:
            species_list = self.species.keys()

        stoichiometry_terms = {species: {} for species in species_list}
        for rxn in self.reactions:
            for reactantId, coefficient in rxn.stoichiometry.items():
                # I forget what types 'coefficient' might be, so 
                # perhaps overzealously I cast it to a float. 
                # Note that this will fail for exotic, non-numerical
                # stoichiometry coefficients. (We could handle these
                # in principle at the cost of detecting them and 
                # using nonlinear conservation constraints.)
                if reactantId in stoichiometry_terms:
                    stoichiometry_terms[reactantId][rxn.id] = float(coefficient)

        for species, terms in stoichiometry_terms.items():
            fid = '_conservation_' + species
            if terms:
                func = Linear(terms, name=fid)
                self.constraints.set(fid, func)

        return stoichiometry_terms

######################################################################
# UTILITY FUNCTIONS

def expression_to_float(expression):
    """Convert a mathematical expression to a floating point, if possible.

    Often we want to identify the constant terms in analytically
    computed derivatives, which in some cases are string expressions
    which contain no variables but can't be parsed as a literal
    floating point value, e.g., '-(1*2)'. This function calls
    em.simplify_expr as necessary to handle these.

    """
    try:
        return np.float(expression)
    except ValueError:
        return np.float(em.simplify_expr(expression))


def tree_sum(terms):

    """Combine terms into a sum parenthesized so as to parse efficiently.

    The expr_manip package used for differentiation in turn uses the
    ast module, which by default parses a sum 'x1 + x2 + ...  + xN' as
    a tree of depth (approximately) N. This raises issues of recursion
    depth and efficiency, which we can (partially) circumvent by
    parenthesizing long sums so that they parse as balanced trees of
    depth approximately log_2 N.

    """
    while len(terms) > 1:
        n = len(terms)
        new_terms = ['(' + terms[i] + ') + (' + terms[i+1] + ')' for
                     i in xrange(0,n-1,2)]
        if i<n-2:
            new_terms.append(terms[n-1])
        terms = new_terms
    
    return terms[0]

