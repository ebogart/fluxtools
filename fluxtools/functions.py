"""Function objects for nonlinear modeling.

Previously we used the SloppyCell FunctionDefinition object to hold
all objective and constraint functions, which worked, although it
diverged completely from the original meaning of the underlying SBML
FunctionDefinition concept. 

Now we want to allow the nonlinear model class to easily treat the
objective function as a constraint or vice versa, for, e.g.,
objective-constrained FVA, without losing expensive-to-recalculate
derivative information, currently maintained separately by the model
for the objective and the constraints. Though we could kludge together
something to do this with the existing code, it seems to make more
sense to adopt an approach in which functions are objects which own
their own derivative information, and may be treated by the model as
either objective or constraint as necessary.

In this version, functions (not the models which use them) own the
compiled code objects for their values and their derivatives. This
will require some revision to the way in which the models supply
namespaces to evaluate the code.

This may also simplify future revision of the function compilation
process.

"""
import re
import expr_manip as em

_word_pattern = re.compile(r'\b[A-Za-z_][\w]*\b')
def translator(dict):
    """ Return function which replaces words in argument according to dict. """
    f = lambda match: dict.get(match.group(), match.group())
    return lambda expression: _word_pattern.sub(f,expression)

class Function:
    def __init__(self, math, variables=None, first_derivatives=None,
                 second_derivatives=None, name=None):
        """Create a mathematical function from an expression string.

        Arguments:
        math - mathematical expression as string. 
        variables - optionally, a set containing all the variables in
            the math expression; if not given these will be extracted
            from the expression automatically
        first_derivatives - optional dict {'x1': dmath_dx1_as_string,
            ...}.  First derivatives not specified in this way will be
            calculated as needed.
        second_derivatives - optional dict {('x1','x2'):
            ddmath_dx1_dx2_as_string, ...}.  Second derivatives not
            specified in this way will be calculated as needed. Key
            tuples (v1,v2) should always be sorted, v1 <= v2.
        name - optional string; has no practical effect, but could be
            useful for tracking where, how, and why the function and
            its associated code objects were created. If not given,
            the expression itself (possibly truncated) will be used
            in the filenames for compiled code objects.

        """
        self.math = math
        if variables is not None:
            self.variables = variables
        else:
            # expr_manip still returns Sets instead of sets; cast the
            # result to set mostly because this annoys me.
            self.variables = set(em.extract_vars(self.math))
        self._first_derivatives = {}
        if first_derivatives:
            self._first_derivatives.update(first_derivatives)
        self._second_derivatives = {}
        if second_derivatives:
            self._second_derivatives.update(second_derivatives)
        self._code = None
        self._first_derivative_code = {}
        self._second_derivative_code = {}
        self.name = name
        if self.name:
            self.tag = name
        elif len(self.math) < 100:
            self.tag = "'%s'" % self.math
        else:
            self.tag = "'%s...'" % self.math[:100]

    def __eq__(self, other):
        return (self.math == other.math)

    def __ne__(self, other):
        return not (self == other)

    def code(self):
        """ Return compiled code object allowing evaluation of this function.

        The code should be eval'd in a namespace providing values for
        all its variables as well as implementations of any 
        relevant functions. 

        """
        if not self._code:
            filename = '<fluxtools function %s>' % self.tag
            self._code = compile(self.math, filename, mode='eval')
        return self._code

    def first_derivative_code(self, variable):
        """Return compiled code object for a derivative of this function.

        No checking is done to ensure this is a sensible variable to
        differentiate this function with respect to; the calling code
        should do such checking to avoid endlessly and expensively
        evaluating zero, constants, etc.

        """
        if variable not in self._first_derivative_code:
            filename = '<derivative of %s wrt %s>' % (self.tag, variable)
            code = compile(self.derivative(variable), filename, mode='eval')
            self._first_derivative_code[variable] = code
        return self._first_derivative_code[variable]

    def second_derivative_code(self, variable_pair):
        """Return code object for a second derivative of this function.

        As with first_derivative_code, arguments are not checked for
        sensibility.

        """
        variable1, variable2 = sorted(variable_pair)
        if (variable1, variable2) not in self._second_derivative_code:
            filename = '<second derivative of %s wrt %s,%s>' % (self.tag, 
                                                                variable1, variable2)
            expr = self.second_derivative((variable1, variable2))
            code = compile(expr, filename, mode='eval')
            self._second_derivative_code[(variable1, variable2)] = code
        return self._second_derivative_code[(variable1, variable2)] 

    def derivative(self, variable):
        """Differentiate this function with respect to variable."""
        cached_derivative = self._first_derivatives.get(variable, None)
        if cached_derivative: 
            return cached_derivative
        else:
            return self._first_derivatives.setdefault(variable,
                                                      em.diff_expr(self.math, 
                                                                   variable))

    def all_first_derivatives(self, set_of_variables=None):
        """Return the nonzero first derivatives of the function. 

        If set_of_variables is given, check and return only derivatives
        with respect to those variables. 

        Returns a dict {'var1': 'derivative_expression_1',
        ...}. Variables absent from the function's math expression are
        not included in the result, so in general the expressions will
        not be '0', but this is not guaranteed: if the function is
        'a-a', for example, the derivative '0' with respect to 'a'
        will be returned.

        """
        if set_of_variables is None:
            subset = self.variables
        else:
            subset = self.variables.intersection(set_of_variables)
        return {v: self.derivative(v) for v in subset}

    def second_derivative(self, variable_pair):
        """Find the second derivative of this function wrt two variables."""
        v1, v2 = sorted(variable_pair)
        cached_derivative = self._second_derivatives.get((v1,v2),None)
        if cached_derivative:
            return cached_derivative
        
        first_derivative = self.derivative(v1)
        return self._second_derivatives.setdefault((v1,v2),
                                                   em.diff_expr(first_derivative,
                                                                v2))

    def all_second_derivatives(self, set_of_variables=None):
        """Return the nonzero second derivatives of the function.

        If set_of_variables is given, consider only second derivatives
        with respect to pairs of those variables. 

        Returns a dict whose keys are sorted tuples of variables
        and whose values are the corresponding second derivative
        expressions. 

        In general only non-'0' terms are included in the result but
        those may appear in cases where variables appear in first
        derivative expressions in such a way that the corresponding
        second derivative is 0.

        """
        if set_of_variables is None:
            subset = self.variables
        else:
            subset = self.variables.intersection(set_of_variables)
        result = {}
        first_derivatives = self.all_first_derivatives(set_of_variables)
        for v1, first_derivative_expression in first_derivatives.iteritems():
            # Extracting all the variables from all the first
            # derivatives every time we call this is no worse than the
            # existing code, which does exactly that.
            # 
            # We only want to evaluate each second derivative once, so 
            # when looking at the first derivative with respect to variable
            # i (numbered in the order resulting from sorting the variables
            # as strings), we consider only second derivatives with 
            # respect to variables i+N,i+(N-1),...,i+1,i. 
            first_derivative_vars = em.extract_vars(first_derivative_expression)
            for v2 in sorted(subset.intersection(first_derivative_vars),
                             reverse=True):
                if v2 < v1:
                    break
                result[(v1,v2)] = self.second_derivative((v1,v2))
        return result

    def substitute(self, substitutions, new_name=None):
        """Make a new function by substituting variables in this function.

        Arguments:
        substitutions - dictionary {old_name_0: new_name_0, old_name_1: ...}
        new_name - name attribute for the new function; if None, the name 
            of this function is used.

        Where keys in the dictionary occur as variables (or
        functions!) in this function's math expression, set of
        variables, or cache of previously calculated derivatives, the
        indicated substitution is applied to obtain a new function,
        which is returned. This function is unchanged.

        This function's caches of compiled code objects, if any, are not 
        transferred to the new function.

        """

        if new_name is None:
            new_name = self.name
        
        substitute = translator(substitutions)
        new_math = substitute(self.math)
        new_variables = {substitute(v) for v in self.variables}
        new_first_derivatives = {substitute(variable): substitute(derivative) for
                                 variable, derivative in self._first_derivatives.iteritems()}
        new_second_derivatives = {tuple(sorted(map(substitute, variable_pair))): 
                                  substitute(second_derivative) for 
                                  variable_pair, second_derivative in 
                                  self._second_derivatives.iteritems()}
        return Function(new_math, variables=new_variables, 
                        first_derivatives=new_first_derivatives, 
                        second_derivatives=new_second_derivatives,
                        name=new_name)
        
            
class Linear(Function):
    def __init__(self, coefficients, name=None):
        """Create a linear combination of variables. 

        Arugments:
        coefficients - dictionary {v1: c1, v2:c2, ...}  where each key
            is a variable (as string) and each value is a number (or
            string representation of a number) giving that variable's
            coefficient in the linear combination.
        name - optional name, no effect

        The coefficients will be converted to strings for math expressions,
        simply calling str(c1), etc.

        """
        coefficients = {v: str(c) for v,c in coefficients.iteritems()}
        expr = ' + '.join('%s*%s' % (c,v) for v,c in coefficients.iteritems())
        Function.__init__(self, expr, variables=set(coefficients),
                          first_derivatives=coefficients, name=name)
        self.coefficients = coefficients
        
    def second_derivative(self, variable_pair):
        return '0'

    def all_second_derivatives(self, set_of_variables=None):
        return {}

    def substitute(self, substitutions, new_name=None):
        """Make a new function by substituting variables in this function.

        Arguments:
        substitutions - dictionary {old_name_0: new_name_0, old_name_1: ...}
        new_name - name attribute for the new function; if None, the name 
            of this function is used.

        Where keys in the dictionary occur as variables (or
        functions!) in this function's math expression, set of
        variables, or cache of previously calculated derivatives, the
        indicated substitution is applied to obtain a new function,
        which is returned. This function is unchanged.

        This function's caches of compiled code objects, if any, are not 
        transferred to the new function.

        """

        if new_name is None:
            new_name = self.name
        
        substitute = translator(substitutions)
        new_coefficients = {substitute(variable): coefficient for
                            variable, coefficient in self.coefficients.iteritems()}

        return Linear(new_coefficients, name=new_name)
