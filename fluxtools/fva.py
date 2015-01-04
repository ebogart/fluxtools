""" Tools for flux variability analysis. """

from Queue import Empty
import numpy as np
import fluxtools.nlcm as nlcm
from fluxtools.nlcm import NonlinearModel
import multiprocessing as mp
from functions import Linear
import logging, pickle

default_n_parallel_procs = 1

def _fva_worker(model, job_queue, result_queue, guess):

    """Worker function for doing FVA with multiprocessing.

    For use as a target of multiprocessing.Process. Each entry in
    job_queue should be a string giving a variable in the model (or
    None, as a termination signal).  The corresponding entry of
    result_queue will be {key: (fva_lower_bound, fva_upper_bound)}
    unless an optimization failure occurs, in which case it will be
    {key: 'failure'}.

    """

    done = 0

    while True:

        try:
            key = job_queue.get(timeout=3600)
        except Empty:
            print 'FVA worker finishing anomalously after completing %d tasks' % done
            return 

        if key is None:
            print 'FVA worker finishing normally after completing %d tasks' % done
            return

        try:
            result = single_fva(model, key, guess)
            result_queue.put({key: result})
        except nlcm.OptimizationFailure:
            result_queue.put({key: 'failure'})
        done += 1


def single_fva(model, variable, guess=None):
    """ Find min/max values of variable in model. 

    The existing objective function is lost. 

    """
    extrema = []
    # Minimize variable, then -1.0*variable 
    # (maximizing it.)
    for sign in (1., -1.):
        model.objective_function = Linear({variable: sign},
                                          name='fva_%s' % variable)
        model.compile()
        if getattr(model, 'repeated_solve_max_attempts', 1) > 1:
            model.repeated_solve(guess, 
                                 model.repeated_solve_max_iter, 
                                 model.repeated_solve_max_attempts)
        else:
            model.solve(guess)
        # Record the flux, not the objective function value
        extrema.append(sign*model.obj_value)
    return extrema

def do_fva(model, variables=None, guess=None,
           n_procs=default_n_parallel_procs, cache={},
           check_failures=True, log_interval=100, log_filename=None):
    """Minimize/maximize (serially) variables in model.

    The model's existing upper/lower bounds are preserved, and
    the existing objective function will be restored after 
    FVA completes. 

    If variables is None, use all variables in the model.

    If cache is given, tuples of extrema will be taken
    from the cache instead of recalculated, wherever 
    possible.

    If n_procs is >1, multiple processes will be spawned to 
    parallelize the FVA process. This may not be faster, if the
    total number of calculations per process is not high.

    Returns a dictionary {variable: (min_possible_value,
                                     max_possible_value) ... }

    If check_failures is True, nlcm.OptimizationFailure is raised
    if any of the individual optimization calculations fail (note, 
    in the parallel case, this is checked only after all have completed.)
    If check_failures is false, variables where either the maximization
    or the minimization step failed will be associated with the string
    'failure' instead of a tuple of bounds.

    If log_filename is given and more than one process is used, every
    log_interval variables an update will be written to log_filename +
    '.txt' and the most recent results will be saved to log_filename +
    '_n.pickle.'

    """
    if log_filename:
        logger = logging.getLogger(log_filename)
        logger.setLevel(logging.INFO)
        fh  = logging.FileHandler(filename=log_filename + '.txt')
        logger.addHandler(fh)
        fh.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        
    if variables is None:
        variables = model.variables

    new_variables = []
    results = {}
    for v in variables:
        if v in cache:
            results[v] = cache[v]
        else:
            new_variables.append(v)

    logging.info('Total FVA results requested: %d' % len(variables))
    logging.info('Found results for old variables: %d' % len(results))
    logging.info('Analyzing new variables: %d' % len(new_variables))
    if not new_variables:
        return results

    original_objective = model.objective_function
    try:
        if n_procs > 1:
            # I think that manually setting up a bunch of worker 
            # processes with information about the model may be faster
            # than using a Pool and providing the model as an argument 
            # each time, though there may be a cleaner way to do this
            # using the tools in the multiprocessing module.
            argument_queue = mp.Queue()
            result_queue = mp.Queue()
            processes = [mp.Process(target=_fva_worker, 
                                    args=(model,
                                          argument_queue, result_queue,
                                          guess)) for i in xrange(n_procs)]
            for v in new_variables:
                argument_queue.put(v)
            # Add termination signals
            for p in processes:
                argument_queue.put(None)
            for p in processes:
                p.start()
            results = {}
            # We won't get them back in order, but we know how many
            # there will be:
            counter = 0 
            counter_max = len(new_variables)
            temp_results = {}
            for v in new_variables:
                result = result_queue.get()
                result_key = result.keys()[0]
                results.update(result)
                if log_filename:
                    temp_results.update(result)
                    if (counter+1) % log_interval == 0:
                        temp_filename = (log_filename +
                                         '_%d.pickle' % counter)
                        with open(temp_filename,'w') as f:
                            pickle.dump(temp_results, f)
                            logger.info('(%d/%d) ' % (counter+1, counter_max) + 
                                        ', '.join(temp_results.keys())) 
                        temp_results = {}
                counter += 1 
            for p in processes:
                p.join()
            failed_variables = [v for v, result in results.iteritems()
                                if result == 'failure']
            if failed_variables and check_failures:
                raise nlcm.OptimizationFailure(
                    'FVA encountered %d optimization failures (%s, ...)' %
                    (len(failed_variables), failed_variables[0])
                )

        else:
            for var in new_variables:
                try:
                    extrema = single_fva(model, var, guess)
                    results[var] = tuple(extrema)
                except nlcm.OptimizationFailure:
                    if check_failures:
                        raise nlcm.OptimizationFailure('FVA failed checking %s' % var)
                    else:
                        results[var] = 'failure'

    finally:
        model.objective_function = original_objective
        model.compile()
    return results

def objective_constrained_fva(model, objective_bounds=None, variables=None, 
                              **kwargs):
    """ Determine extreme variable values consistent with objective value.

    Arguments:
    model - instance to test
    objective_bounds - scalar or tuple; exact value/lower and upper limits
        to impose on objective function in FVA calculation. If None, 
        sets the objective function to model.obj_value.
    variables - variables to test; if None, all variables
    **kwargs - optional arguments to pass to do_fva
    
    """
    constraint_id='_fva_objective_constraint'
    if constraint_id in model.constraints.keys():
        raise ValueError('Bad default constraint name.')
    model.constraints.set(constraint_id, model.objective_function)

    if objective_bounds is None:
        objective_bounds = model.obj_value
    model.set_bound(constraint_id, objective_bounds)

    try:
        fva_result = do_fva(model, variables=variables, **kwargs)
    finally:
        model.remove_constraint(constraint_id) 
        model.compile()
    return fva_result
