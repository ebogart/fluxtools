import re
import numpy as np
from nlcm import NonlinearModel
from functions import Function, Linear

def enclose(function, variable, name=None):
    """Given a function f(x0, x1...), write a function f(x0, x1...) - variable.

    For example, if the argument function corresponds to 'x**2 + sin(y) + c',
    the return function will correspond to '(x**2 + sin(y) + c) - variable'.

    This is useful if, for instance, we have a model with a constraint
    or objective function g(x0, x1...) and need to introduce a new
    variable which is constrained to be equal to the value of g.

    If the argument function is linear (that is, an instance of
    functions.Linear,) so is the return function.

    Otherwise, the first and second derivative caches are preserved,
    and the first derivative wrt variable is filled in.

    """
    if isinstance(function, Linear):
        new_coefficients = {'variable': -1}
        new_coefficients.update(function.coefficients)
        return Linear(new_coefficients, name=name)

    new_expr = '(%s) - %s' % (function.math, variable)
    new_variables = function.variables.copy()
    new_variables.add(variable)
    new_function = Function(new_expr, variables=new_variables,
                            first_derivatives=function._first_derivatives,
                            second_derivatives=function._second_derivatives,
                            name=name)
    new_function._first_derivatives[variable] = '-1.'
    return new_function

class ReplicaModel(NonlinearModel):
    
    """Optimization problem built from many replicas of a base problem.
    
    For the elastic band method, etc.

    """

    def __init__(self, base_model, N, split_parameters=(), save_objectives=True):

        """Create a new optimization problem from images of a base problem.

        If save_objectives is true, the base model's objective function
        will be preserved for each image as a variable constrained to 
        equal that image's objective function value, and a default 
        objective function which is the sum of the individual objective
        functions will be applied. 

        The replica model is not compiled by default.

        The parameters of the base problem will not be split across images,
        except for those listed in split_parameters.

        """

        NonlinearModel.__init__(self)
        # Compile the base problem to ensure all its derivative
        # caches are populated and its variable list is up-to-date.
        base_model.compile()
        self.base_variables = base_model.variables
        _base_variable_set = set(self.base_variables)

        # Parameters are constant across all images.
        shared_parameters = {k: v for k,v in
                             base_model.parameters.iteritems() if k
                             not in split_parameters}
        self.parameters = shared_parameters

        # Set up a template for constraints requiring
        # variables to equal the objective value for each image
        if save_objectives:
            template = enclose(base_model.objective_function, 
                               'objective_value')
            # Note enclose() will copy
            # the derivative caches automatically, 
            # and they will be preserved by template.substitute() 
            # below.
            objectives_by_image = []

        for i in xrange(N):
            name_map = {n: self.metaname(n,i) for n in 
                        self.base_variables}
            name_map.update({p: self.metaname(p,i) for p in
                             split_parameters})

            # Initialize local parameters
            self.parameters.update({name_map[p]: base_model.parameters[p]
                                    for p in split_parameters})

            # Translate all constraints
            for key, g0 in base_model.constraints.items():
                new_name = self.metaname(g0.name, i)
                new_key = self.metaname(key, i)
                gi = g0.substitute(name_map, new_name)
                self.constraints.set(new_key, gi)
                self.set_bound(new_key, base_model.get_bounds(key))

            # Apply all variable bounds
            for v in self.base_variables:
                self.set_bound(self.metaname(v,i), base_model.get_bounds(v))

            # Set up a new variable equal to this image's original 
            # objective function. 
            if save_objectives:
                this_objective = self.metaname('objective_value',i)
                objectives_by_image.append(this_objective)
                name_map['objective_value'] = this_objective
                new_id = self.metaname('objective_constraint', i)
                local_constraint = template.substitute(name_map, new_id)
                self.constraints.set(new_id, local_constraint)
                self.set_bound(new_id, 0.)

        # Set the combined objective function, if desired
        if save_objectives:
            coefficients = dict.fromkeys(objectives_by_image, 1.)
            self.objective_function = Linear(coefficients, 
                                             name = 'combined_objective_function')
                
        self.N = N

    def set_global_bound(self, base_variable, bound):
        """ Set bounds on one base variable/constraint across all images. """
        for i in xrange(self.N):
            self.set_bound(self.metaname(base_variable,i), bound)

    def set_global_bounds(self, base_variable_bounds):
        """ Set bounds on many base-model variables/constraints from dict. """
        for base_v, bound in base_variable_bounds.iteritems():
            self.set_global_bound(base_v, bound)

    def metaname(self, name, imagenumber):
        """ Return the translation in this model of 'name' in an image. """
        return 'image%d_%s' % (imagenumber, name)

    def basename(self, name):
        """ Return the base-model translation of 'name' in this model, if any.

        """
        if name.startswith('image'):
            return name.split('_',1)[1]
        else:
            return None

    def resolve_name(self, descriptor):
        """ Allow variable v in image i to be accessed as (i,v). """
        if isinstance(descriptor, tuple):
            n, variable = descriptor
            return self.metaname(variable, n)
        else:
            return descriptor

    def solve(self, x0=None, options={}):
        NonlinearModel.solve(self, x0, options=options)

        self.soln_by_image = dict((i, 
                                   dict((var, 
                                         self.soln[self.metaname(var, i)]) 
                                        for var in self.base_variables)) 
                                  for i in
                                  xrange(self.N))
        self.soln_global = {k:v for k,v in self.soln.iteritems()
                            if not k.startswith('image')}
        self.x_by_image = np.array([[self.soln_by_image[i][v] for 
                                     v in self.base_variables]
                                    for i in xrange(self.N)]).T

