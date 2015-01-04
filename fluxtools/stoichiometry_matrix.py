import numpy as np
import scipy.sparse as sp
import glpk # as provided by pyglpk; other modules of same name will differ!

def sdict_from_sloppy_net(net):
    ''' 
    Extract a stoichiometry dictionary from a SloppyCell ReactionNetwork
    object. 

    Returns a dictionary mapping the id attribute of each reaction in
    the network to that reaction's stoichiometry attribute, which is 
    a dictionary of the reaction's stoichiometric coefficients, by
    species id.
    '''
    return dict([(r.id, r.stoichiometry) for r in net.reactions])

def sdict_from_SloppyCell_by_name(net):
    ''' 
    Extract a stoichiometry dictionary from a SloppyCell ReactionNetwork
    object. 

    Differs from sdict_from_sloppy_net in that it tries to 
    generate a dictionary which refers to every reaction and species
    in the network by its name attribute, not its id attribute. Where simple
    human-readable names have been munged or replaced to comply with the
    SBML ID standards, this can be a great convenience. However, names are
    not guaranteed to be unique or even specified at all. This function will 
    check that each Reaction and Species object in the network has 
    a unique name, raising an exception if not.

    Returns a dictionary mapping the name attribute of each reaction in
    the network to a dictionary of the reaction's stoichiometric coefficients,
    keyed by species name.
    '''
    reaction_names = [r.name for r in net.reactions if r.name]
    species_names = [s.name for s in net.species if s.name]
    name_set = set([])
    name_set.update(reaction_names)
    name_set.update(species_names)
    if len(name_set) != (len(net.reactions) + len(net.species)):
        raise Exception('Missing or non-unique reaction or species name(s)' + 
                        ' in SloppyCell network.')
    
    species_id_to_name = dict([(s.id,s.name) for s in net.species])
    sdict = {}
    for r in net.reactions:
        stoichiometry = dict([(species_id_to_name[s], c) for 
                              s,c in r.stoichiometry.items()])
        sdict[r.name] = stoichiometry

    return sdict

class StoichiometryMatrix (object): 
    ''' Provides a useful interface for a large stoichiometry matrix 
    represented in a scipy.sparse.lil_matrix. It's a bit ad hoc at the moment.

    Though we anticipate that the entries in the matrix will be integers,
    they are stored as floats by default. 

    Note an lil_matrix is chosen as it is cheap to change its sparsity 
    pattern. Arithmetic operations may call for one of the other scipy
    sparse matrix types, while setting up the linear programming
    problem may best be done from a coordinate matrix.

    Ultimately it might be best to use coo_matrix for everything
    and set up an initialization process which constructs coordinate
    lists rather than changing individual entries-- however that might
    not support slicing/indexing as well; I can't remember if I've implemented
    anything that relies on slicing yet (deletion was going to, but that
    could be done instead  in a sparsity-aware way.)
    '''

    def __init__(self, stoichiometry_dictionary):
        self.num_reactions = 0 
        self.num_compounds = 0
        self.reactions = []
        self.compounds = []
        if stoichiometry_dictionary is None:
            # The array itself will be built from nothing by repeated
            # addition of rows and columns. 
            self.S = None
            raise Exception('Initialization of empty matrix is not supported yet.')

        else:
            # Preallocate the array for faster initialization.
            self.reactions = stoichiometry_dictionary.keys()
            compound_set = set([compound 
                                for s in stoichiometry_dictionary.values()
                                for compound in s.keys()])
            self.compounds = list(compound_set)
            self.num_reactions = len(self.reactions)
            self.num_compounds = len(self.compounds)
            compound_index = dict(zip(self.compounds, 
                                      range(self.num_compounds)))

            self.S = sp.lil_matrix((self.num_compounds, self.num_reactions))

            for column_number, reaction in enumerate(self.reactions):
                stoichiometry = stoichiometry_dictionary[reaction]
                for compound in stoichiometry:
                    self.S[compound_index[compound], column_number] = stoichiometry[compound] 
                    

    def add_reaction(self, reaction, stoichiometry):
        ''' Attempting to add a reaction already in the matrix will update
        that reaction's stoichiometry in place. '''
        if reaction not in self.reactions:
            self.reactions.append(reaction)

            new_column = sp.lil_matrix((self.num_compounds, 1))
            if self.S is None:
                self.S  = new_column
            else:
                self.S = sp.hstack([self.S, new_column], format = 'lil') 

            reaction_index = self.num_reactions
            self.num_reactions += 1

        else:
            reaction_index = self.reactions.index(reaction)
            
        for compound in stoichiometry.keys():
            compound_index = self.add_compound(compound)
            self.S[compound_index, reaction_index] = stoichiometry[compound]


    def add_compound(self, compound):
        '''Add a compound to the matrix's list of compounds and add a 
        corresponding row of zeros to the matrix itself, returning the index
        of that row. (If the compound already has a row in the matrix, add
        nothing and return the index of that row.)''' 
        try: 
            return self.compounds.index(compound)

        except ValueError:
            self.compounds.append(compound)
            new_row = sp.lil_matrix((1, self.num_reactions))
            if self.S is None:
                self.S = new_row
            else: 
                self.S = np.vstack([self.S, new_row], format = 'lil')

            self.num_compounds += 1
            return self.num_compounds - 1

    def delete_reaction(self, reaction):
        pass

    def delete_compound(self, compound):
        pass


    def add_transport_reaction(self, compound):
        # This may be the only place so far where I've assumed that the 
        # compounds are strings.
        reaction_label = 'transport_' + compound
        # Import and export are equivalent unless reversibility is a problem
        self.add_reaction(reaction_label, {compound: 1})


#    def delete_compound_and_associated_reactions(self, compound):
#        pass


    def column_vector_from_dict(self, compound_dict):
        v = np.zeros((self.num_compounds, 1))
        for compound in compound_dict:
            v[self.compounds.index(compound)] = compound_dict[compound]

        return v

    def column_list_from_dict(self, compound_dict, default=0.0):
        v = [default] * self.num_compounds
        for compound in compound_dict:
            v[self.compounds.index(compound)] = compound_dict[compound]

        return v

    def row_vector_from_dict(self, reaction_dict):
        v = np.zeros((self.num_reactions, 1))
        for reaction in reaction_dict:
            v[self.reactions.index(reaction)] = reaction_dict[reaction]

        return v

    def row_list_from_dict(self, reaction_dict, default=0.0):
        v = [default] * self.num_reactions
        for reaction in reaction_dict:
            v[self.reactions.index(reaction)] = reaction_dict[reaction]

        return v

    def annotate_column_vector(self, vector):
        if len(vector) == self.num_compounds:
            labelled_list = [(self.compounds[i], float(v)) 
                             for i, v in enumerate(vector) 
                             if float(v)]
            return dict(labelled_list)
        else:
            raise Exception('Column length must match the number of compounds in the matrix.')

    def annotate_row_vector(self, vector):
        if len(vector) == self.num_reactions:
            labelled_list = [(self.reactions[i], float(v)) 
                             for i, v in enumerate(vector) 
                             if float(v)]
            return dict(labelled_list)
        else:
            raise Exception('Row length must match the number of reactions in the matrix.')


class ConstraintModel(StoichiometryMatrix):

    ''' A general linear constraint model class. It extends 
    StoichiometryMatrix and handles creation and solution of linear
    programming problems through glpk. '''

    def __init__(self, stoichiometry = None, maximize = False):
#                 other_constraints = {}):
        '''
        Create a linear constraint model with specified stoichiometry.

        Arguments:
        stoichiometry -- a reaction-stoichiometry dictionary
        maximize -- if true, maximize the objective function when
        solving the linear programming problem (default false)
        other_constraints -- optional, additional linear constraints on the 
        variables of the model (reaction rates), specified as a dictionary 
        of the form 
        {'constraint_name': ({'variable1': coefficient1,
                              'variable2': coefficient2, ...},
                              right_hand_side), ...}
        to indicate that coefficient1*variable1 + coefficient2*variable2 must
        equal right_hand side, etc.
        
        '''
        if stoichiometry is None or isinstance(stoichiometry, dict):
            StoichiometryMatrix.__init__(self, stoichiometry)
        elif isinstance(stoichiometry, StoichiometryMatrix):
            raise Exception('Initialization from stoichiometry matrix not implemented yet.')
        else:
            raise Exception('Wrong data type for constraint model initialization.')

        self.other_constraints = {}

        # right_hand_side: bounds on the entries of S * flux_vector.
        # Bounds may be numbers, None, or duples of those types 
        # (lower_bound, upper_bound).
        # 
        # Typically the right hand side will enforce a steady state 
        # constraint , self.S * vector = 0
        self.right_hand_side = [0.0] * self.num_compounds
   
        # flux_bounds: restrictions on the entries of the flux vector.
        # Bounds may be numbers, None, or duples of those types 
        # (lower_bound, upper_bound).
        # Initially let all fluxes be unbounded.
        self.flux_bounds = [None] * self.num_reactions

        # self.maximize: True if we are maximizing the objective function;
        # False if we are minimizing it
        self.maximize = maximize

        # There is no good default choice for the objective function, but 
        # initializing all objective coefficients to zero is convenient
        self.objective = [0.0] * self.num_reactions
        
        # self.lp will be the pyglpk linear programming problem object 
        # created when self.solve() is called. Note it will not be updated
        # to reflect changes in the state of the model between calls to 
        # self.solve().
        self.lp = None 

#    def set_compound_bounds
    def set_all_compound_bounds(self, compound_bounds):
        ''' Set the right-hand side from a vector or compound-bound
        dictionary.''' 
        try:
            self.right_hand_side = compound_bounds[0:self.num_compounds]
        except TypeError:
            self.right_hand_side = self.column_list_from_dict(compound_bounds)

#    def set_flux_bounds
    def set_all_flux_bounds(self, flux_bounds):
        ''' Set the limits on reaction fluxes from a vector or 
        reaction-bound dictionary.'''
        try: 
            self.flux_bounds = flux_bounds[0:self.num_reactions]
        except TypeError:
            self.flux_bounds = self.row_list_from_dict(flux_bounds,
                                                       default = None)

#    def set_objective_coefficients
    def set_objective_function(self, objective):
        ''' Set the objective coefficients of each reaction from a vector
        or reaction-coefficient dictionary. '''
        try: 
            self.objective = objective[0:self.num_reactions]
        except TypeError:
            self.objective = self.row_list_from_dict(objective)

    # Getters?
            
    def set_ratio(self, reaction_a, reaction_b, ratio):
        ''' 
        Enforce (or relax) the constraint reaction_a/reaction_b = ratio.

        Add ({reaction_a: -1.0, reaction_b: ratio}, 0.) to 
        self.other_constraints, under the key '_ratio_%s_%s' % (reaction_a,
        reaction_b). If ratio is None, remove the entry in 
        self.other_constraints with that key, if any. 

        This facilitates the common operation of adding a generic
        linear constraint to fix the ratio of two variables.

        '''

        key = '_ratio_%s_%s' % (reaction_a, reaction_b)
        
        if ratio is None:
            self.other_constraints.pop(key, None)
        
        else:
            self.other_constraints[key] = ({reaction_a: -1.0,
                                            reaction_b: ratio},
                                           0.)

    def add_reaction(self, reaction, stoichiometry, **constraints):
        ''' If reaction is not already in the model, add it. Optional arguments
        flux_bound and objective_coefficient set those values for the new
        reaction, which default to None and 0.0 respectively. If reaction
        is in the model, update its stoichiometry (and flux bound and 
        objective coefficient if those arguments are present) in place.
        '''
        ## Design note: ultimately both this and the parent class 
        ## should have an add_reaction function which raises an exception
        ## if the reaction is already present and an update_reaction function
        ## which does not.

        # In any case, as I'm discovering while changing self.S to a
        # scipy.sparse.lil_matrix, it is awkward to have child classes
        # referring to S directly! This is something to do for the future.

        if reaction not in self.reactions:
            self.reactions.append(reaction)

            new_column = sp.lil_matrix((self.num_compounds, 1))
            if self.S is None:
                self.S = new_column
            else:
                self.S = sp.hstack([self.S, new_column], format = 'lil')

            self.flux_bounds.append(None)
            self.objective.append(0.0)

            reaction_index = self.num_reactions
            self.num_reactions += 1

        else:
            reaction_index = self.reactions.index(reaction)
            
        for compound in stoichiometry.keys():
            compound_index = self.add_compound(compound)
            self.S[compound_index, reaction_index] = stoichiometry[compound]

        if 'objective_coefficient' in constraints:
            self.objective[reaction_index] = constraints['objective_coefficient']
        if 'flux_bound' in constraints:
            self.flux_bounds[reaction_index] = constraints['flux_bound']
            
    def remove_reaction(self, reaction):
        ''' Remove a reaction from the model. Compounds which were involved
        only in that reaction will remain in the model with zero rows 
        in the stoichiometry matrix, which may be undesirable. Deleting the
        only reaction in a model is currently an exception.'''
        
        reaction_index = self.reactions.index(reaction)
        
        if self.num_reactions == 1:
            raise Exception("Can't remove only reaction in model.") 

        # Remove the corresponding column from the stoichiometry matrix
        if reaction_index == 0:
            self.S = self.S[:,1:]
        elif reaction_index == self.num_reactions - 1:
            self.S = self.S[:,:-1]
        else: 
            self.S = sp.hstack([self.S[:,:reaction_index],
                                self.S[:,(reaction_index + 1):]],
                                format = 'lil')
        
        self.flux_bounds.pop(reaction_index)
        self.objective.pop(reaction_index)
        self.reactions.pop(reaction_index)
            
        self.num_reactions -= 1

    def add_compound(self, compound):
        '''Add a compound to the matrix's list of compounds and add a 
        corresponding row of zeros to the matrix itself, returning the index
        of that row. (If the compound already has a row in the matrix, add
        nothing and return the index of that row.)''' 
        try: 
            return self.compounds.index(compound)

        except ValueError:
            self.compounds.append(compound)
            new_row = sp.lil_matrix((1, self.num_reactions))
            if self.S is None:
                self.S = new_row
            self.S = sp.vstack([self.S, new_row], format = 'lil')
            self.right_hand_side.append(0.0)
            self.num_compounds += 1
            return self.num_compounds - 1

    # Eventually, will need to overload remove_compound, etc. 


    def do_not_conserve(self, *compounds):
        ''' Do not enforce conservation of these compounds: set the 
        right-hand-side of the equations corresponding to each 
        compound to None. '''
        for compound in compounds:
            if compound in self.compounds:
                self.right_hand_side[self.compounds.index(compound)] = None

    def update_lp_problem(self):
        ''' Creates a GLPK linear programming problem instance self.lp from 
        the current state of the constraint model.'''
        lp = glpk.LPX()
        self.lp = lp

        lp.obj.maximize = self.maximize

        lp.rows.add(self.num_compounds + len(self.other_constraints))
        lp.cols.add(self.num_reactions)
        
        # Impose constraints arising from conservation of each compound
        for i,c in enumerate(self.compounds):
            lp.rows[i].name = c
            lp.rows[i].bounds = self.right_hand_side[i]

        # Impose miscellaneous other constraints if any, placing them
        # at the end of the list of rows of the linear problem.
        # We make a local copy of the constraints as a list, so we 
        # know we have a consistent ordering throughout this function.
        other_constraints = self.other_constraints.items() 
        iter_other_constraints = enumerate(other_constraints,
                                           start = self.num_compounds)
        for index, (name, (coefficients, bounds)) in iter_other_constraints:
            lp.rows[index].name = name
            lp.rows[index].bounds = bounds

        for c in lp.cols:
            c.name = self.reactions[c.index]
            c.bounds = self.flux_bounds[c.index]

        lp.obj[:] = self.objective

        constraint_matrix = self.S.tocoo()
        # PyGLPK will not recognize numpy int32/int64 as valid 
        # integers for matrix indexing purposes, so must cast the 
        # row and column indexes of the coo_matrix explicitly
        rows = map(int, constraint_matrix.row)
        cols = map(int, constraint_matrix.col)
        data = constraint_matrix.data

        # Next we must add to this matrix rows corresponding to the
        # assorted other constraints. We assume these will be few 
        # compared to the compounds in the system, so it is not 
        # crucial to do this efficiently.
        # First, reset the iterator:
        iter_other_constraints = enumerate(other_constraints,
                                           start = self.num_compounds)
        additional_rows = []
        additional_cols = []
        additional_data = []
        for row_index, (n, (coefficients, bounds)) in iter_other_constraints:
            for reaction,coefficient in coefficients.iteritems():
                column_index = self.reactions.index(reaction)
                additional_rows.append(row_index)
                additional_cols.append(column_index)
                additional_data.append(coefficient) 

        lp.matrix = zip(rows,cols,data) + zip(additional_rows,
                                              additional_cols,
                                              additional_data)

    def solve(self):
        ''' Creates a GLPK linear programming problem instance self.lp from 
        the current state of the constraint model and attempts to solve it
        with the simplex method. '''
        
        self.update_lp_problem()
        self.lp.simplex()
        fluxes = dict([(c.name, c.primal) for c in self.lp.cols])
        
#        print "Optimization status:", self.lp.status
#        print "Objective function value:", self.lp.obj.value

        return fluxes

    def impose_default_bounds(self, lower, upper):
        ''' Impose the specified upper bound on those reactions that
        have no upper bound and the specified lower bound on those 
        that have no lower bound. For use when arbitrary bounds on otherwise
        unbounded reactions are necessary, as in sampling, eg.
        '''
        for i in range(self.num_reactions):
            if self.flux_bounds[i] == None:
                self.flux_bounds[i] = (lower, upper)
            elif isinstance(self.flux_bounds[i],tuple):
                bounds = list(self.flux_bounds[i])
                if bounds[0] == None:
                    bounds[0] = lower
                if bounds[1] == None:
                    bounds[1] = upper
                self.flux_bounds[i] = tuple(bounds)

    def random_objective_sample(self, npoints, tolerance=1e-19):
        """ Sample vertices of the flux cone, more or less.

        Find a series of flux vectors optimizing random objective
        functions. In a non-rigorous and likely biased sense, this
        samples the vertices of the allowed flux cone. (Compare the
        process of generating warmup points for the COBRA toolbox's ACHR
        sampler.) All fluxes must be bounded, and the bounds must allow
        at least one solution; nonfeasible bounds or the presence of
        unbounded optimal solutions will lead to errors. The model's
        existing maximization flag and objective function are ignored in
        the sampling process, but returned to their original state after
        sampling is finished.
        
        Returns a list of npoints dense ndarrays of shape
        (num_reactions,) where containing those elements of each flux
        vector which were larger (in magnitude) than the specified
        tolerance.

        Currently this will not lead to sensible results for a
        NonNegativeConstraintModel because of the way objective
        coefficients for reversible reactions are handled there.

        """

        original_maximize = self.maximize
        original_objective = self.objective

        samples = [] #sp.lil_matrix((self.num_reactions, npoints))
        reaction_index = dict(zip(self.reactions,range(self.num_reactions)))

        self.maximize = True

        for i in range(npoints):
            v = np.random.randn(self.num_reactions)
            v = v/np.linalg.norm(v)
            # Maximize projection onto unit vectors
            # pointing in random directions in reaction space
            self.objective = list(v)
            self.update_lp_problem()
            self.lp.simplex()
            if self.lp.status == 'opt':
                # It might be appropriate to use a sparse array here...
                flux = np.array([c.primal for c in self.lp.cols])
                flux = flux * (np.abs(flux)>tolerance)
                samples.append(flux)
            else: 
                self.maximize = original_maximize
                self.objective = original_objective
                raise Exception('Optimization failure; check that bounds are properly specified and feasible.')

        self.maximize = original_maximize
        self.objective = original_objective

        return samples


    def orthogonal_objective_sample(self, tolerance=1e-19):
        ''' Find flux vectors optimizing an orthogonal set of objective 
        functions seeking to maximize/minimize the flux through each 
        reaction in the model. This should non-randomly sample
        the vertices of the allowed flux cone. (Compare 
        the process of generating warmup points for the COBRA toolbox's
        ACHR sampler.) All fluxes must be bounded, and the bounds must
        allow at least one solution; nonfeasible bounds or the presence of
        unbounded optimal solutions will lead to errors. The model's
        existing maximization flag and objective function are ignored in
        the sampling process, but returned to their original state 
        after sampling is finished. 
        
        Returns a sparse matrix (lil_matrix of shape (num_reactions, npoints)) 
        containing those elements of each flux vector which were larger
        (in magnitude) than the specified tolerance, as well as a dictionary
        of tuples with the upper and lower bounds for the flux through 
        each reaction (subject to the prior flux bounds on that reaction!)

        Currently this will not lead to sensible results for a 
        NonNegativeConstraintModel because of the way objective coefficients
        for reversible reactions are handled there.
        '''

        original_maximize = self.maximize
        original_objective = self.objective

        samples = sp.lil_matrix((self.num_reactions, 2*self.num_reactions))
        reaction_index = dict(zip(self.reactions,range(self.num_reactions)))

        achievable_bounds = {}

        for i in range(self.num_reactions):
            self.objective = [0.0] * self.num_reactions
            self.objective[i] = 1.0 

            self.maximize = True
            self.update_lp_problem()
            self.lp.simplex()
            max_status = self.lp.status
            # This is inefficient but I forget whether the lp.cols
            # are ordered in the same way as self.reactions; check this
            # and replace as necessary
            for c in self.lp.cols:
                if np.abs(c.primal) > tolerance:
                    samples[reaction_index[c.name],2*i] = c.primal 

            self.maximize = False
            self.update_lp_problem()
            self.lp.simplex()
            min_status = self.lp.status
            for c in self.lp.cols:
                if np.abs(c.primal) > tolerance:
                    samples[reaction_index[c.name],2*i+1] = c.primal 
                    
            if max_status != 'opt' or min_status != 'opt':
                self.maximize = original_maximize
                self.objective = original_objective
                raise Exception('Optimization failure (minimization status: ' 
                                + min_status + ', maximization status: ' +
                                max_status + ');' + 
                                ' check that bounds are properly specified and feasible.')
            achievable_bounds[self.reactions[i]] = (samples[i,2*i],
                                                    samples[i,2*i+1])

        self.maximize = original_maximize
        self.objective = original_objective

        return samples, achievable_bounds


class NonNegativeConstraintModel(ConstraintModel):

    ''' As a constraint model, but require all fluxes to be positive 
    at the level of the GLPK model (allowing 'path-distance' type objective
    functions, which minimize the total flux) while still allowing 
    single reactions to be treated as reversible at the ConstraintModel
    level. 

    This change should be mostly transparent; unlike previous versions,
    scalar objective coefficients for reversible reactions work normally
    (that is, setting a coefficient of c for a reversible reaction r
    adds terms c*forward_r_rate + -1.0*c*reverse_r_rate to the 
    objective function as represented internally.)
    
    Separate objective coefficients for the forward and reverse
    parts of a reversible reaction may be specified by setting the 
    reaction's objective coefficient to a tuple of the form 
    (foward_coefficient, reverse_coefficient.) 

    If a tuple of coefficients is specified for an irreversible reaction
    the reverse coefficient will be ignored.

    Other modifications:

    - The model object now has a list of booleans called reversible, 
    of length num_reactions. (Future versions will dispense with this
    and assume the user knows what they are doing when specifying 
    a negative lower bound on a reaction rate.)
    - Add_reaction now has a reversible option, defaulting to false.
    Reactions supplied through __init__() default to irreversible; this
    may be overridden by setting the argument 'reversible' to True or to 
    a list of booleans as long as the reaction dictionary indicating the
    reversibility of each.
    - Flux bounds for reversible reactions work as expected.
    - An irreversible reaction will have minimum flux zero if its minimum
    flux is set to a sub-zero value, behaving normally otherwise.

    '''

    def __init__(self, stoichiometry = None, maximize = False, 
                 reversible = True, reverse_tag = 'reverse_'):
     
        ConstraintModel.__init__(self, stoichiometry, maximize)
        if reversible == False:
            self.reversible = [False] * self.num_reactions
        elif reversible == True:
            self.reversible = [True] * self.num_reactions
        else: 
            self.reversible = reversible

        self.reverse_tag = reverse_tag

    def add_reaction(self, reaction, stoichiometry, 
                     reversible=False, **constraints):
        # This needlessly repeats the membership test when it's already
        # being done in the parent function.
        if reaction not in self.reactions:
            self.reversible.append(reversible)
        else:
            self.reversible[self.reactions.index(reaction)] = reversible 

#        super(NonNegativeConstraintModel,self).add_reaction(reaction, 
#                                                            stoichiometry, 
#                                                            **constraints)
        ConstraintModel.add_reaction(self, reaction, 
                                     stoichiometry, 
                                     **constraints)

    def remove_reaction(self,reaction):
        self.reversible.pop(self.reactions.index(reaction))
        ConstraintModel.remove_reaction(self, reaction)

    def reverse_name(self,reaction):
        ''' Returns the appropriate name for the reverse of a reaction.

        Presently this is simply self.reverse_tag + reaction.
        '''

        return self.reverse_tag + reaction
    def unreverse_name(self,reaction):
        ''' 
        Given a reverse reaction name, return the forward reaction name.

        Presently this is the reverse reaction name with self.reverse_tag
        removed from wherever it first occurs (if anywhere.)
        
        '''
        return reaction.replace(self.reverse_tag,'',1)

    def is_reverse_name(self,reaction):
        return reaction.startswith(self.reverse_tag)
        
    def update_lp_problem(self):
        ''' 
        Creates a GLPK linear programming problem instance self.lp from 
        the current state of the constraint model.

        '''
        lp = glpk.LPX()
        self.lp = lp

        lp.obj.maximize = self.maximize

        num_reversible = sum(self.reversible) # Yes, you can sum booleans

        lp.rows.add(self.num_compounds + len(self.other_constraints))
        lp.cols.add(self.num_reactions + num_reversible)
        
        for index, compound in enumerate(self.compounds):
            r = lp.rows[index]
            r.name = compound
            r.bounds = self.right_hand_side[index]

        # Impose miscellaneous other constraints if any, placing them
        # at the end of the list of rows of the linear problem.
        # We make a local copy of the constraints as a list, so we 
        # know we have a consistent ordering throughout this function.
        other_constraints = self.other_constraints.items() 
        iter_other_constraints = enumerate(other_constraints,
                                           start = self.num_compounds)
        for index, (name, (coefficients, bounds)) in iter_other_constraints:
            lp.rows[index].name = name
            lp.rows[index].bounds = bounds


        # Give the objective function the correct length
        lp.obj[:] = [0] * len(lp.cols)

        reverse_reaction_count = 0
        reverse_reaction_index = {}

        for c in lp.cols[0:self.num_reactions]:
            c.name = self.reactions[c.index]
            bounds = self.flux_bounds[c.index]
            if not isinstance(bounds, tuple):
                bounds = (bounds, bounds)

            if bounds[0] >= 0.0:
                forward_lowerbound = bounds[0]
                reverse_upperbound = 0.0
            elif bounds[0] == None:
                forward_lowerbound = 0.0
                reverse_upperbound = None
            else: 
                forward_lowerbound = 0.0
                reverse_upperbound = -1.0 * bounds[0]

            if bounds[1] == None or bounds[1] >= 0.0:  
                forward_upperbound = bounds[1] 
                reverse_lowerbound = 0.0
            else:
                forward_upperbound = 0.0
                reverse_lowerbound = -1.0 * bounds[1]

            c.bounds = (forward_lowerbound, forward_upperbound)

            if isinstance(self.objective[c.index], tuple):
                forward_coefficient = self.objective[c.index][0]
                reverse_coefficient = self.objective[c.index][1]
            else: 
                forward_coefficient = self.objective[c.index]
                reverse_coefficient = -1.0*self.objective[c.index]

            lp.obj[c.index] = forward_coefficient

            if self.reversible[c.index]:
                reverse_reaction_index[c.index] = (reverse_reaction_count + 
                                                   self.num_reactions)
                reverse_column = lp.cols[reverse_reaction_index[c.index]]
                reverse_column.name = 'reverse_' + str(c.name)
                reverse_column.bounds = (reverse_lowerbound, 
                                         reverse_upperbound)
                lp.obj[reverse_reaction_index[c.index]] = reverse_coefficient

                reverse_reaction_count += 1

        # Conceptually, we set up an expanded stoichiometric matrix 
        # whose final columns represent the reverse direction of the
        # reversible reactions; if we were working with dense arrays, 
        # this could be done as follows: 
        # reversible_columns = self.S[:,np.array(self.reversible)] 
        # expanded_matrix = np.hstack([self.S, reversible_columns])
        # lp.matrix = expanded_matrix.flat
        # Sparse matrices (at least lil_matrix objects) don't support 
        # indexing with logical arrays (and must be iterated over 
        # differently). 
                
        constraint_matrix = self.S.tocoo()
        # PyGLPK will not recognize numpy int32/int64 as valid 
        # integers for matrix indexing purposes, so must cast the 
        # row and column indexes of the coo_matrix explicitly
        rows = map(int, constraint_matrix.row)
        cols = map(int, constraint_matrix.col)
        # The following line appears to have been a complete error, leading
        # to many bugs:
        #data = map(int, constraint_matrix.data)
        data = constraint_matrix.data

        # Next we must add to this matrix rows corresponding to the
        # assorted other constraints. 
        # First, reset the iterator:
        iter_other_constraints = enumerate(other_constraints,
                                           start = self.num_compounds)
        additional_rows = []
        additional_cols = []
        additional_data = []
        for row_index, (n, (coefficients, bounds)) in iter_other_constraints:
            for reaction,coefficient in coefficients.iteritems():
                column_index = self.reactions.index(reaction)
                additional_rows.append(row_index)
                additional_cols.append(column_index)
                additional_data.append(coefficient) 

        constraint_matrix = zip(rows,cols,data) + zip(additional_rows,
                                                      additional_cols,
                                                      additional_data)

        reversible_matrix = []
        for i, j, d in constraint_matrix:
            if self.reversible[j]:
                reversible_matrix.append( (i, reverse_reaction_index[j],
                                           -1.0*d) ) 
        lp.matrix = constraint_matrix + reversible_matrix

    def solve(self, full_output = False):
        ''' 
        Creates a GLPK linear programming problem instance self.lp from 
        the current state of the constraint model and attempts to solve it
        with the simplex method. 

        '''
        
        self.update_lp_problem()
        self.lp.simplex()
        fluxes = dict([(c.name, c.primal) for c in self.lp.cols])
        
        print "Optimization status:", self.lp.status
        print "Objective function value:", self.lp.obj.value

        if not full_output:
            forward_fluxes = {}
            for r,f in fluxes.iteritems():
                if self.is_reverse_name(r):
                    f = -1.0*f
                    r = self.unreverse_name(r)
                forward_fluxes[r] = forward_fluxes.get(r,0.) + f
            fluxes = forward_fluxes

        return fluxes


    def make_irreversible(self, reaction):
        reaction_index = self.reactions.index(reaction)
        if self.reversible[reaction_index]:
            self.reversible[reaction_index] = False
            if isinstance(self.objective[reaction_index], tuple):
                self.objective[reaction_index] = self.objective[reaction_index][0]

    def make_reversible(self,reaction):
        self.reversible[self.reactions.index(reaction)] = True

    
    def objective_function_all_ones(self):
        ''' 
        Set the objective function to extremize total flux.

        All reactions, including both the forward and reverse parts 
        of reversible reactions, will receive an objective coefficient
        of 1.0. 
        
        '''
        self.objective = [1.0] * self.num_reactions
        for i, reversible in enumerate(self.reversible):
            if reversible:
                self.objective[i] = (1.0, 1.0)


    # The following utilities could probably be incorporated into 
    # the parent class.
    def add_transport_reaction(self, compound):
        ''' Add a reversible import-export reaction for the compound. '''
        # This may be the only place so far where I've assumed that the 
        # compounds are strings.
        reaction_label = 'transport_' + compound
        self.add_reaction(reaction_label, {compound: 1}, reversible=True)    
        return reaction_label

    def add_import_reaction(self, compound):
        ''' Add an irreversible import reaction for the compound. '''
        reaction_label = 'import_' + compound
        self.add_reaction(reaction_label, {compound: 1}, reversible=False) 
        return reaction_label

    def add_export_reaction(self, compound):
        ''' Add an irreversible export reaction for the compound. '''
        reaction_label = 'export_' + compound
        self.add_reaction(reaction_label, {compound: -1}, reversible=False)    
        return reaction_label


def conservation_check(model, external_transport_reactions, 
                       test_species = None,
                       threshold = 1e-14):
    ''' Set the flux through the external reactions to zero and 
    determine whether any compounds can then be created from nothing
    (or consumed without production of any other compounds) by 
    the remaining reactions. The original flux bounds on the transport
    reactions, objective function, maximization flag, and right-hand-side 
    may all be overwritten in this process. 

    Species which may be created/destroyed in a process which 
    concomitantly creates/destroys other species (eg, by the reaction
    (nothing) -> A + B) are not distinguished from those which may be 
    created/destroyed in isolation.

    Returns a list of species which may be created by the model
    and a list of species which may be destroyed by the model.

    If test_species is given, check only those species; otherwise, check
    the entire set of species in the model. Note that, when a list of
    test_species is still provided, only created/destroyed species 
    from that list will be returned, even if their creation/destruction
    requires the simultaneous creation/destruction of other species.

    Assumes the model does not contain a reaction '_conservation'. '''

    for r in external_transport_reactions:
        model.flux_bounds[model.reactions.index(r)] = 0.0

    model.objective = [0.0] * model.num_reactions
    model.maximize = True

    if test_species == None: 
        test_species = model.compounds[:]

    # Check for creation of species

    potentially_created_species = test_species[:]
    created_species = []
    
    model.right_hand_side = [(0.0, None)] * model.num_compounds

    for s in test_species: #model.compounds:
        if s in potentially_created_species:

            print 'Checking ' + s + '.'
            model.add_reaction('_conservation',{s: -1},
                               objective_coefficient = 1.0,
                               flux_bound = (0.0,None))

            f = model.solve()
            potentially_created_species.remove(s)

            if model.lp.status == 'opt' and model.lp.obj.value > threshold:

                created_species.append(s)

                for matrix_row in model.lp.rows:
                    if ((matrix_row.value > threshold) and
                        (matrix_row.name in potentially_created_species)):
                        created_species.append(matrix_row.name)
                        potentially_created_species.remove(matrix_row.name)

            elif model.lp.status == 'unbnd':
                created_species.append(s)

            else:
                if model.lp.status not in ['opt', 'unbnd']:
                    raise Exception('Optimization failure checking compound' +
                                    str(s))

            model.remove_reaction('_conservation')

    # Check for destruction of species
                            
    potentially_destroyed_species = test_species[:]
    destroyed_species = []
    
    model.right_hand_side = [(None, 0.0)] * model.num_compounds

    print 'Entering species destruction test.'

    for s in test_species: #model.compounds:
        if s in potentially_destroyed_species:

            print 'Checking ' + s + '.'
            model.add_reaction('_conservation',{s: 1},
                               objective_coefficient = 1.0,
                               flux_bound = (0.0,None))

            potentially_destroyed_species.remove(s)
            f = model.solve()

            if model.lp.status == 'opt' and model.lp.obj.value > threshold:

                destroyed_species.append(s)

                for matrix_row in model.lp.rows:
                    if ((matrix_row.value < -1.0*threshold) and
                        (matrix_row.name in potentially_destroyed_species)):
                        destroyed_species.append(matrix_row.name)
                        potentially_destroyed_species.remove(matrix_row.name)

            elif model.lp.status == 'unbnd':
                destroyed_species.append(s)

            else:
                if model.lp.status not in ['opt', 'unbnd']:
                    raise Exception('Optimization failure checking compound' +
                                    str(s))
            

            model.remove_reaction('_conservation')

    return created_species, destroyed_species

def discrete_conservation_check(model, external_transport_reactions, 
                                test_species = None,
                                threshold = 1e-14):
    ''' Set the flux through the external reactions to zero and 
    determine whether any compounds can then be created from nothing
    (or consumed without production of any other compounds) by 
    the remaining reactions. The original flux bounds on the transport
    reactions, objective function, and maximization flag
    may all be overwritten in this process; the existing right-hand side
    is respected.

    Species must be created or destroyed in isolation, ie, the 
    reaction '(none) -> A + B' will not be considered to create 'A'
    without some way to destroy 'B'. (Compare to conservation_check,
    above.)

    Returns a list of species which may be created by the model
    and a list of species which may be destroyed by the model.

    If test_species is given, check only those species; otherwise, check
    the entire set of species in the model.

    Assumes the model does not contain a reaction '_conservation'. '''

    for r in external_transport_reactions:
        model.flux_bounds[model.reactions.index(r)] = 0.0

    model.objective = [0.0] * model.num_reactions
    model.maximize = True

    if test_species == None: 
        test_species = model.compounds[:]

    # Check for creation of species

    potentially_created_species = test_species[:]
    created_species = []
    
    for s in test_species: #model.compounds:
        if s in potentially_created_species:

            print 'Checking ' + s + '.'
            model.add_reaction('_conservation',{s: -1},
                               objective_coefficient = 1.0,
                               flux_bound = (0.0,None))

            f = model.solve()
            potentially_created_species.remove(s)

            if model.lp.status == 'opt' and model.lp.obj.value > threshold:
                created_species.append(s)

            elif model.lp.status == 'unbnd':
                created_species.append(s)

            else:
                if model.lp.status not in ['opt', 'unbnd']:
                    raise Exception('Optimization failure checking compound' +
                                    str(s))

            model.remove_reaction('_conservation')

    # Check for destruction of species
                            
    potentially_destroyed_species = test_species[:]
    destroyed_species = []

    print 'Entering species destruction test.'

    for s in test_species: #model.compounds:
        if s in potentially_destroyed_species:

            print 'Checking ' + s + '.'
            model.add_reaction('_conservation',{s: 1},
                               objective_coefficient = 1.0,
                               flux_bound = (0.0,None))

            potentially_destroyed_species.remove(s)
            f = model.solve()

            if model.lp.status == 'opt' and model.lp.obj.value > threshold:
                destroyed_species.append(s)

            elif model.lp.status == 'unbnd':
                destroyed_species.append(s)

            else:
                if model.lp.status not in ['opt', 'unbnd']:
                    raise Exception('Optimization failure checking compound' +
                                    str(s))
            
            model.remove_reaction('_conservation')

    return created_species, destroyed_species


def nonconserving_reaction_counter(model, external_transport_reactions, 
                                   test_species = None,
                                   threshold = 1e-14):
    ''' Set the flux through the external reactions to zero and 
    determine whether any compounds can then be created from nothing
    (or consumed without production of any other compounds) by 
    the remaining reactions. The original flux bounds on the transport
    reactions, objective function, and maximization flag
    may all be overwritten in this process; the existing right-hand side
    is respected.

    Species must be created or destroyed in isolation, ie, the 
    reaction '(none) -> A + B' will not be considered to create 'A'
    without some way to destroy 'B'. (Compare to conservation_check,
    above.)

    Returns dictionaries mapping species that may be created by the model
    and species that may be destroyed by the model to lists of the 
    reactions carrying flux in the process found to create or 
    destroy them (or to None, if the process was unbounded; limitation
    of the solver as I understand it.) 

    If test_species is given, check only those species; otherwise, check
    the entire set of species in the model.

    Assumes the model does not contain a reaction '_conservation'. '''

    for r in external_transport_reactions:
        model.flux_bounds[model.reactions.index(r)] = 0.0

    model.objective = [0.0] * model.num_reactions
    model.maximize = True

    if test_species == None: 
        test_species = model.compounds[:]

    # Check for creation of species

    potentially_created_species = test_species[:]
    created_species = {}
    
    for s in test_species: #model.compounds:
        if s in potentially_created_species:

            print 'Checking ' + s + '.'
            model.add_reaction('_conservation',{s: -1},
                               objective_coefficient = 1.0,
                               flux_bound = (0.0,None))

            f = model.solve()
            potentially_created_species.remove(s)

            if model.lp.status == 'opt' and model.lp.obj.value > threshold:
                created_species[s] = [r for r in f if np.abs(f[r]) > threshold]

            elif model.lp.status == 'unbnd':
                created_species[s] = None

            else:
                if model.lp.status not in ['opt', 'unbnd']:
                    raise Exception('Optimization failure checking compound' +
                                    str(s))

            model.remove_reaction('_conservation')

    # Check for destruction of species
                            
    potentially_destroyed_species = test_species[:]
    destroyed_species = {}

    print 'Entering species destruction test.'

    for s in test_species: #model.compounds:
        if s in potentially_destroyed_species:

            print 'Checking ' + s + '.'
            model.add_reaction('_conservation',{s: 1},
                               objective_coefficient = 1.0,
                               flux_bound = (0.0,None))

            potentially_destroyed_species.remove(s)
            f = model.solve()

            if model.lp.status == 'opt' and model.lp.obj.value > threshold:
                destroyed_species[s] = [r for r in f if np.abs(f[r]) > 
                                        threshold]

            elif model.lp.status == 'unbnd':
                destroyed_species[s] = None

            else:
                if model.lp.status not in ['opt', 'unbnd']:
                    raise Exception('Optimization failure checking compound' +
                                    str(s))
            
            model.remove_reaction('_conservation')

    return created_species, destroyed_species


def nonnegative_nonconserver_count(model, external_transport_reactions, 
                                   test_species = None,
                                   threshold = 1e-14,
                                   check_destruction = True):
    ''' Find conservation violations in a NonNegativeConstraintModel.

    Test the model to see whether each species may be created or destroyed 
    (possibly as part of multi-species combinations) and record the reactions
    involved in the conservation-violation process as well as the 
    total flux, using the NNCM's total-flux capability to avoid cycles and
    generally simplify the solutions, which may reveal problematic reactions
    more clearly.

    Note that the objection, transport reaction flux bounds, and right hand
    side of the model will be overwritten. Flux bounds for reactions other 
    than the specified transport reactions will be preserved and respected
    in the checking process.

    Arguments:
    model -- NonNegativeConstraintModel instance to test
    external_transporters -- set of reactions to suppress while testing. If 
    all transporters are balanced (using a set of external species, etc., 
    it is not strictly necesssary to do this.)

    Keyword arguments:
    test_species -- collection of species whose conservation to check; if None,
    test all species in the model (the default behavior).
    threshold -- threshold below which flux absolute values are treated as zero
    check_destruction -- if true, check for destruction as well as creation
    of species (the default.) Skipping that check will save time if, 
    eg, all reactions are reversible and it is thus redundant.

    Returns:

    created_species -- list containing, for each species which may be created,
    a tuple of its name, the list of reactions implicated in the creation
    process, and the minimal total flux necessary to achieve unit flux 
    through a sink reaction for the species.
    destroyed_species (if requested) -- analogous list for species which may
    be destroyed
    
    '''

    for r in external_transport_reactions:
        model.flux_bounds[model.reactions.index(r)] = 0.0

    model.objective = [(1.0, 1.0)] * model.num_reactions
    model.maximize = False

    if test_species == None: 
        test_species = model.compounds[:]

    # Check for creation of species

    potentially_created_species = test_species[:]
    created_species = []

    model.right_hand_side = [(0.0, None)] * model.num_compounds
    
    for s in test_species: #model.compounds:
        if s in potentially_created_species:

            print 'Checking ' + s + '.'
            model.add_reaction('_conservation',{s: -1},
                               objective_coefficient = 0.0,
                               flux_bound = 1.0)

            f = model.solve()
            # Don't record the flux through the source/sink explicitly
            f.pop('_conservation') 
            potentially_created_species.remove(s)

            if model.lp.status == 'opt':
                created_species.append((s, 
                                        [(r[8:] if r.startswith('reverse_') 
                                          else r) 
                                         for r in f if 
                                         np.abs(f[r]) > threshold],
                                        model.lp.obj.value))
            # This ought not to occur
            elif model.lp.status == 'unbnd':
                created_species[s] = None

            else:
                if model.lp.status not in ['opt', 'unbnd', 'nofeas']:
                    raise Exception('Optimization failure checking compound' +
                                    str(s))

            model.remove_reaction('_conservation')


    if not check_destruction:
        return created_species

    # Check for destruction of species
                            
    potentially_destroyed_species = test_species[:]
    destroyed_species = []

    print 'Entering species destruction test.'

    model.right_hand_side = [(None, 0.0)] * model.num_compounds

    for s in test_species: #model.compounds:
        if s in potentially_destroyed_species:

            print 'Checking ' + s + '.'
            model.add_reaction('_conservation',{s: 1},
                               objective_coefficient = 0.0,
                               flux_bound = 1.0)

            potentially_destroyed_species.remove(s)
            f = model.solve()
            f.pop('_conservation') 

            if model.lp.status == 'opt':
                destroyed_species.append((s, 
                                          [(r[8:] if r.startswith('reverse_') 
                                            else r) 
                                           for r in f if 
                                           np.abs(f[r]) > threshold],
                                          model.lp.obj.value))
                
            # This ought not to occur either
            elif model.lp.status == 'unbnd':
                destroyed_species[s] = None

            else:
                if model.lp.status not in ['opt', 'unbnd', 'nofeas']:
                    raise Exception('Optimization failure checking compound' +
                                    str(s))
            
            model.remove_reaction('_conservation')

    return created_species, destroyed_species

def model_from_net(
        net,
        default_bound = 1000.,
        extras={}, 
        extra_bounds={},
        non_conserved=set(),
        do_conserve=[],
        free_compartments=('biomass',
                           'external')):
    """ Create an NNCM instance from a network object. 

    Arguments:
    net - network object to convert
    default_bound - value to use for upper bound on
       reaction fluxes (inc. of rates of reverse reactions,
       for reversible reactions)
    extras - dictionary of reactions to add to the model
       and their stoichiometries
    bounds - bounds on the reactions in extras
    non_conserved - species to exempt from the Sv=0 
       constraint 
    do_conserve - species to conserve even if they are in 
       non_conserved or are in one of the free_compartments
    free_compartments - list of compartments the species in 
       which should be exempted from the Sv=0 constraint
       (defaults to ('biomass','external')). Note that 
       species not present in net, but appearing in the
       stoichiometries of an extra reaction, will not 
       be considered to belong to these compartments
       even if they obey the usual species_compartment
       naming scheme.
    
    Returns:
    m - NonNegativeConstraintModel instance
    bounds - dict of bounds on the reactions of m
    objective - the objective function of m, defaulting to 
        a coefficient of (1,1) for all reactions (that is,
        a minimum-total-flux objective).
    non_conserved - the final set of species not conserved in 
        m. 

    """
    all_stoichiometries = sdict_from_sloppy_net(net)
    all_stoichiometries.update(extras)
    bounds = dict.fromkeys(all_stoichiometries,
                           (0., default_bound))
    bounds.update(
        dict.fromkeys(
            (r.id for r in net.reactions if 
             r.reversible),
            (-1.0*default_bound, default_bound)))
    bounds.update(extra_bounds)
    objective = dict.fromkeys(
        all_stoichiometries,
        (1., 1.))
    all_species = set()
    for stoichiometry in all_stoichiometries.values():
        all_species.update(stoichiometry)
    non_conserved = non_conserved.copy()
    non_conserved.update({s for s in all_species if s in 
                          net.species.keys() and 
                          net.species.getByKey(s).compartment in
                          free_compartments})

    for s in do_conserve:
        if s in non_conserved:
            non_conserved.remove(s)
    m = NonNegativeConstraintModel(all_stoichiometries)
    m.set_all_flux_bounds(bounds)
    m.set_objective_function(objective)
    m.do_not_conserve(*non_conserved)
    return m, bounds, objective, non_conserved
