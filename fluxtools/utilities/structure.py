""" Tools for examining the local structure of an optimization problem. """

from scipy.sparse import coo_matrix
import numpy as np

def list_fixed_variables(model):
    fixed_variables = {v for v in model.variables if model.get_upper_bound(v) ==
                       model.get_lower_bound(v) and model.get_upper_bound(v) is
                       not None}
    return fixed_variables

def jg(m, ignore_fixed=True):
    jac_g = np.array(coo_matrix((m.eval_jac_g(m.x, False),
                                 m.eval_jac_g(m.x, True),),
                                shape = (m.ncon, m.nvar)).todense())
    if ignore_fixed:
        print m.nvar
        print jac_g.shape
        fixed_variables = list_fixed_variables(m)
        nonfixed_indices = [i for i,v in enumerate(m.variables) if v not in
                            fixed_variables]
        print np.max(nonfixed_indices)
        jac_g = jac_g[:,nonfixed_indices]
            
    return jac_g

def row(i, m): 
    tol = 1e-7
    jac_g = jg(m, ignore_fixed=False)
    print 'Gradient entry %d (variable %s): %.4f' % (i, m.variables[i],
                                                     m.eval_grad_f(m.x)[i])
    print 'equals'
    total = 0.
    for j in range(m.ncon):
        lambda_j = m.constraint_multipliers[j]
        jac_entry = jac_g[j,i]
        if np.abs(lambda_j) > tol and np.abs(jac_entry) > tol:
            print '%.4g * %.4g\t(%s)' % (lambda_j, jac_entry, 
                                         m.constraints[j].name)
            total += lambda_j * jac_entry

    if np.abs(m.zl[i]) > tol:
        print '-1.0 * %.4g\t(%s lower bound)' % (m.zl[i], 
                                                m.variables[i])
        total += -1.0*m.zl[i]
    if np.abs(m.zu[i]) > tol: # SIGN MAY BE WRONG HERE
        print '1.0 * %.4g\t(%s upper bound)' % (m.zu[i], 
                                                m.variables[i])
        total += m.zu[i]
    print 'total %.4g' % total
    return total

def active_matrix(m, ignore_fixed=True):
    fixed_variables = list_fixed_variables(m)
    nonfixed_variables = [v for v in m.variables if v not in 
                         fixed_variables]
    tol = 1e-7
    cons_cols = jg(m, ignore_fixed=ignore_fixed)
    if ignore_fixed:
        var_cols = np.eye(len(nonfixed_variables))
    else:
        var_cols = np.eye(m.nvar)
    active_cols = []
    active_ids = []
    x = m.x
    g = m.eval_g(x)
    gl, gu = m.make_constraint_bound_vectors()
    xl, xu = m.make_variable_bound_vectors()
    def close(bound, value):
        return (bound is not None and np.abs(bound-value) < tol)
    for i, (value, lb, ub) in enumerate(zip(g, gl, gu)):
        if close(lb, value) or close(ub, value):
            active_cols.append(cons_cols[i])
            active_ids.append(m.constraints[i].name)
    for i, (value, lb, ub) in enumerate(zip(x, xl, xu)):
        variable = m.variables[i]
        if ignore_fixed and variable in fixed_variables:
            continue
        if close(lb, value) or close(ub, value):
            if ignore_fixed:
                active_cols.append(var_cols[nonfixed_variables.index(variable)])
            else:
                active_cols.append(var_cols[i])
            active_ids.append(variable)
    M = np.vstack(active_cols).T
    return M, active_ids

def active_redundant(m):
    tol = 1e-7
    M, ids = active_matrix(m)
    l = []
    r = np.diag(np.linalg.qr(M, mode='r'))
    for name, entry in zip(ids, r):
        if np.abs(entry) < tol:
            l.append(name)
    return l

def show_scaling(model, x, n=10, ignore_fixed=True):
    l1 = zip(np.abs(model.eval_grad_f(x)), model.variables)
    jgi = zip(*model.eval_jac_g(x, True))
    l2 = [(f, model.constraints[t[0]].name or model.constraints[t[0]].tag or 'constraint %d' % t[0],
           model.variables[t[1]])
           for f, t in zip(np.abs(model.eval_jac_g(x,False)), jgi)]
    l = sorted(l1 + l2, reverse=True)
    if ignore_fixed:
        fixed_variables = list_fixed_variables(model)
        l = [t for t in l if t[-1] not in fixed_variables]
    return l[:n]

