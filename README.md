fluxtools
=========

fluxtools sets up and solves flux balance analysis problems with nonlinear constraints. Utilities for creating large-scale models from copies of a base model
are also included.

Usage
-----

Create a network object by loading a metabolic model from sbml:

    import fluxtools.sbml_interface as si
    net = si.fromSBMLfile('example.xml')

or from a dictionary of reaction stoichiometries:

    import fluxtools.reaction_networks as rn
    stoichiometries = {'tx_A': {'A_ext': -1., 'A': 1.},
                       'R1': {'A': -1, 'B': 1},
                       'R2': {'A': -1, 'C': 1},
                       'R3': {'B': -1, 'D': 1},
                       'R4': {'C': -1, 'D': 1},
                       'sink': {'D': -1}}
    net = rn.net_from_dict(stoichiometries)

Then, convert it to an optimization model:

    from fluxtools.nlcm import NonlinearNetworkModel
    model = NonlinearNetworkModel('example', net,
                                  external_species_ids=('A_ext',))
    model.set_objective('max_sink', '-1.0*sink')

Note the objective function is minimized. The basic FBA steady-state
constraints are incorporated automatically (specified external species
are not conserved.) 

Add additional constraints (at least one nonlinear constraint must
be given):

    model.add_constraint('quadratic', 'R2-R1**2', 0.)
    model.compile() 
    
Set upper and lower bounds on variables:

    model.set_bound('R3', (0., 2.))

Solve, and interact with the solution:

    >>> model.solve() # optionally, specify an initial guess as numpy array
    [... IPOPT output omitted...]
    array([ 2.00000002,  4.00000008,  2.        ,  4.00000008,  6.0000001 ,
            6.0000001 ])
    >>> model.obj_value
    -6.0
    >>> model.soln['R2']
    4.0

The model may be adjusted and re-solved, but `model.compile()` should
be called every time a constraint is added or removed. 

Installation
------------

The package itself can simply be installed with `python setup.py install`
once all dependencies are available.

Dependencies
------------

* Numpy
* Scipy
* nose
* libsbml python bindings, http://sbml.org/Software/libSBML
* pyipopt (recommended version: http://github.com/ebogart/pyipopt)
* IPOPT, https://projects.coin-or.org/Ipopt
* pyglpk, http://tfinley.net/software/pyglpk/ (change '-m32' options in setup.py to '-m64' if necessary and be prepared to ignore some irrelevant failing tests with recent versions of glpk)
* glpk, https://www.gnu.org/software/glpk/

Testing
-------

Tests of much of the package's functionality are included. Run 
'nosetests' in the source directory to run the tests.

Organization
------------

* nlcm - nonlinear optimization models with IPOPT
* stoichiometry_matrix - strictly linear FBA models with GLPK
* sbml_interface - read and write models in SBML format
* gene_utilities - load and save (some) GPR associations
* deblock - identify and purge blocked reactions and singleton 
  metabolites based on the structure of the metabolic network
* fluxmap - trace which reactions consume/produce individual 
  species in a solution
* c4clone - duplicate a network and connect the copies, as when
  setting up bundle sheath and mesophyll cells in C4 plants
* expr_manip - differentiation and parsing tools
* reaction_networks - provides classes representing models and their
  components 
* simplification - simplify models by examining their linear subproblems
* functions - classes for representing constraint/objective functions
* keyedlist - the KeyedList data structure
* utilities - assorted tools for dealing with COBRA-compliant SBML files, 
  total flux objectives, and analyzing optimization problem structure
* test - test suite

Documentation
-------------
A (slightly) more detailed introduction for users and notes on numerical
considerations are in preparation.

Author
------
Eli Bogart, elb87@cornell.edu

Acknowledgments
---------------

Code for the expr_manip, sbml_interface, keyedlist, and reaction_networks modules originated in SloppyCell, http://sloppycell.sourceforge.net/




