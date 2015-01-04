""" Reaction network model class, primarily for SBML import

This has been drastically pared down from the SloppyCell version and
it may eventually become desirable to put some of that code back in! 

In particular the _makeCrossReferences method and the lists of variables
of different types (assigned, algebraic, optimizable, etc.,) could
be quite useful, though it would need to be adjusted to consider
reaction rates as first-class variables.

"""
from __future__ import division
# This makes integer like one would expect mathematically, e.g. 1/2 = .5
#  rather than 0. This is obviously safer when loading other folks' models.
#  It does, however, cost about 10% performance on PC12.
# This is expected to become standard behavior around python 3.0

import copy
import sets
import types
import time
import os
import sys
import operator

import logging
logger = logging.getLogger('reaction_networks.Network_mod')

import scipy
import math

from .. import keyedlist, expr_manip
KeyedList = keyedlist.KeyedList

import Reactions
from Components import *

class Network:

    def __init__(self, id, name=''):
        self.id, self.name = id, name

        self.functionDefinitions = KeyedList()
        self.reactions = KeyedList()
        self.assignmentRules = KeyedList()
        self.rateRules = KeyedList()
        self.algebraicRules = KeyedList()
        self.events = KeyedList()
        self.constraints = KeyedList()
        self.compartments = KeyedList()
        self.parameters = KeyedList()
        self.species = KeyedList()        


    #
    # Methods used to build up a network
    #

    def add_compartment(self, id, initial_size=1.0, name='', 
                        typical_value=None,
                        is_constant=True, is_optimizable=False):
        """Add a compartment to the Network.

        All species must reside within a compartment.

        """
        compartment = Compartment(id, initial_size, name, typical_value, 
                                  is_constant, is_optimizable)
        self._checkIdUniqueness(id)
        self.compartments.set(id,compartment)

    def add_species(self, id, compartment, initial_conc=0, 
                    name='', typical_value=None,
                    is_boundary_condition=False, is_constant=False, 
                    is_optimizable=False, uniprot_ids=None):
        """Add a species to the Network.

        """
        species = Species(id, compartment, initial_conc, name, typical_value,
                          is_boundary_condition, is_constant, is_optimizable,
                          uniprot_ids)
        self._checkIdUniqueness(id)
        self.species.set(id,species)

    def add_parameter(self, id, initial_value=1.0, name='',
                      typical_value=None,
                      is_constant=True, is_optimizable=True):
        """Add a parameter to the Network.

        """
        parameter = Parameter(id, initial_value, name, is_constant, 
                              typical_value, is_optimizable)
        self._checkIdUniqueness(id)
        self.parameter.set(id,parameter)

    def add_event(self, id, trigger, event_assignments={}, delay=0, name='',
                  buffer=0):
        """Add an event to the Network.

        id - id for this event
        trigger - The event firest when trigger passes from False to True.
            Examples: To fire when time becomes greater than 5.0:
                       trigger = 'gt(time, 5.0)'
                      To fire when A becomes less than sin(B/C):
                       trigger = 'lt(A, sin(B/C))'
        event_assignments - A dictionary or KeyedList of assignments to make
                      when the event executes.
            Example: To set A to 4.3 and D to B/C
                      event_assignments = {'A': 4.3,
                                           'D': 'B/C'}
        delay - Optionally, assignments may take effect some time after the
            event fires. delay may be a number or math expression
        name - A more detailed name for the event, not restricted to the id
            format

        """
        event = Event(id, trigger, event_assignments, delay, name,
                      buffer)
        self._checkIdUniqueness(event.id)
        self.events.set(event.id, event)

    def add_constraint(self, id, trigger, message=None, name=''):
        """Add a constraint to the Network.

        id - id for this constraint
        trigger - We treat constraints as events that correspond to an
            invalid solution whenever the trigger is True.
            Example: To have an invalid solution when species A is > 5.0:
                       trigger = 'lt(A, 5.0)'
        name - A more detailed name for the constraint, not restricted
            to the id format

        """
        constraint = ConstraintEvent(id, trigger, message, name)
        self._checkIdUniqueness(constraint.id)
        self.constraints.set(constraint.id, constraint)

    def add_func_def(self, id, variables, math, name=''):
        """Add a function definition to the Network.

        id - id for the function definition
        variables - The variables used in the math expression whose
                    values should be subsituted.
        math - The math expression the function definition represents
        name - A more extended name for the definition

        Example:
            To define f(x, y, z) = y**2 - cos(x/z)
            net.add_func_def('my_func', ('x', 'y', 'z'), 'y**2 - cos(x/z)')

        """
        func = FunctionDefinition(id, variables, math, name)
        self._checkIdUniqueness(func.id)
        self.functionDefinitions.set(func.id, func)

    def addReaction(self, id, *args, **kwargs):
        # Reactions can be added by (1) passing in a string representing
        #  kinetic law, or (2) passing in a class already specifying the 
        #  kinetic law.
        # XXX: I'm a little unhappy with this because option (2) breaks the
        #      pattern that the first argument is the id
        if isinstance(id, str):
            rxn = apply(Reactions.Reaction, (id,) + args, kwargs)
        else:
            rxn = apply(id, args, kwargs)

        self._checkIdUniqueness(rxn.id)
        self.reactions.set(rxn.id, rxn)

    def add_assignment_rule(self, var_id, rhs, index=None):
        """Add an assignment rule to the Network.

        A rate rules species that <var_id> = rhs.

        index: Optionally specify which index in the list of rules
               should be used for the new rule. This is important (in
               principle) because assignment rules must be evaluated
               in order. The default is to add to the end of the list.

        """
        self.set_var_constant(var_id, False)
        if index is None:
            self.assignmentRules.set(var_id, rhs)
        else:
            self.assignmentRules.insert_item(index, var_id, rhs)

    def add_rate_rule(self, var_id, rhs):
        """Add a rate rule to the Network.

        A rate rules species that d <var_id>/dt = rhs.

        """
        self.set_var_constant(var_id, False)
        self.rateRules.set(var_id, rhs)

    def add_algebraic_rule(self, rhs):
        """Add an algebraic rule to the Network.

        An algebraic rule specifies that 0 = rhs.

        """
        self.algebraicRules.set(rhs, rhs)

    def remove_component(self, id):
        """Remove the component with the given id from the Network.

        Components that can be removed are variables, reactions, events,
        function definitions, assignment rules, rate rules, and constraints.

        """
        complists = [self.species, self.parameters, self.compartments,
                     self.reactions, self.functionDefinitions,
                     self.events, self.constraints,
                     self.assignmentRules, self.rateRules,
                     self.algebraicRules]
        for complist in complists:
            # If the id is in a list and has a non-empty name
            if complist.has_key(id):
                complist.remove_by_key(id)

    def _checkIdUniqueness(self, id):
        """Check whether a given id is already in use by this Network.

        """
        if id == 'time':
            logger.warn("Specifying 'time' as a variable is dangerous! Are you "
                        "sure you know what you're doing?")
        elif id == 'default':
            logger.warn("'default' is a reserved keyword in C. This will cause "
                        "problems using the C-based integrator.")
        elif id[0].isdigit():
            raise ValueError("The id %s is invalid. ids must not start with a "
                             "number." % id)
        # Should this check the various types of rules as well?
        if id in self.species.keys()\
           or id in self.compartments.keys()\
           or id in self.parameters.keys()\
           or id in self.reactions.keys()\
           or id in self.functionDefinitions.keys()\
           or id in self.events.keys()\
           or id in self.constraints.keys()\
           or id == self.id:
            raise ValueError, ('The id %s is already in use!' % id)

    def set_id(self, id):
        """Set the id of this Network

        """
        if id != self.id:
            self._checkIdUniqueness(id)
            self.id = id

    def get_id(self):
        """Get the id of this Network.

        """
        return self.id

    def set_name(self, name):
        """Set the name of this Network

        """
        self.name = name 

    def get_name(self):
        """Get the name of this Network.

        """
        return self.name

    def copy(self, new_id=None, new_name=None):
        """Return a copy of the given network, with an optional new id.

        """
        new_net = copy.deepcopy(self)
        if new_id is not None:
            new_net.set_id(new_id)
        if new_name is not None:
            new_net.set_name(new_name)

        return new_net

    def __getstate__(self):
        # deepcopy automatically does a deepcopy of whatever we return
        #  here, so we only need to do a shallow copy and remove functions 
        odict = copy.copy(self.__dict__)
        return odict

    def __setstate__(self, newdict):
        self.__dict__.update(newdict)

    def get_component_name(self, id, TeX_form=False):
        """Return a components's name if it exists, else just return its id.

        """
        # These are all the things that have names (except the network itself)
        complists = [self.reactions, self.functionDefinitions,
                     self.events, self.constraints, self.species,
                     self.compartments, self.parameters]
        # If we don't find a name, we'll just use the id
        name = id
        for complist in complists:
            # If the id is in a list and has a non-empty name
            if complist.has_key(id) and complist.get(id).name:
                name = complist.get(id).name
                break

        # We can also check the network's name
        if id == self.id and self.name:
            name = self.name

        if TeX_form:
            # If we've got one underscore in the name, use that to indicate a 
            #  subscript
            if name.count('_') == 1:
                sp = name.split('_')
                name = '%s_{%s}' % (sp[0], sp[1])
            else:
                # TeX can't handle more than one _ in a name, so we substitute
                name = name.replace('_', r'\_')

        return name

def net_from_dict(stoichiometry_dictionary, name='automatic'):
    """ Create a network from a dictionary of reaction stoichiometries.

    The network will have one compartment ('main'). Species will
    be inferred from stoichiometry entries.

    """
    net = Network(name)
    net.add_compartment('main')
    all_species = set()
    for reaction, stoichiometry in stoichiometry_dictionary.iteritems():
        for species in stoichiometry:
            all_species.add(species)
        new_reaction = Reactions.Reaction(reaction, stoichiometry,
                                          name=reaction, kineticLaw=None)
        net.reactions.set(reaction, new_reaction)
    for species in all_species:
        net.add_species(species, 'main')
    return net

