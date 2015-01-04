import os
import atexit
import AST
import Differentiation
import Extraction
import Py2TeX
import Simplify
import Substitution

from AST import strip_parse, ast2str
from Differentiation import load_derivs, save_derivs, diff_expr
from Extraction import extract_vars, extract_funcs, extract_comps
from Py2TeX import expr2TeX
from Simplify import simplify_expr
from Substitution import sub_for_var, sub_for_func, sub_for_vars, sub_for_comps
from Substitution import make_c_compatible 

from .. import _TEMP_DIR

# We load a dictionary of previously-taken derivatives for efficiency
load_derivs(os.path.join(_TEMP_DIR, 'diff.pickle'))

# This will save the diffs dictionary upon exit from the python interpreter.
atexit.register(save_derivs, os.path.join(_TEMP_DIR, 'diff.pickle'))
