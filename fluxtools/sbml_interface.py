""" Load/save models to/from SBML.

Much code copied from SloppyCell.ReactionNetworks.SBMLInterface

The loading method allows loading a SloppyCell reaction network from a minimal,
possibly corrupt or invalid SBML file. This is a pared down version of
SloppyCell's normal from_SBML_string method. It is useful for loading
constraint-based models, which may have no kinetic laws, and models
which may have certain deficiencies in standards compliance, eg empty
kineticLaw containers.

This code ignores the following SBML component types, which we 
cannot currently use:
- Event
- Constraint

TO DO: fix loading of global parameters, saving of local parameters. 
SloppyCell used one homogeneous parameter namespace by munging local
parameter names appropriately and saved everything as a global parameters.

"""
import os, sys, HTMLParser
import libsbml


import expr_manip
from reaction_networks import Network_mod
from keyedlist import KeyedList

# sbml_level and sbml_version are default parameters to pass to
# constructors for libsbml 4.0 (and later)
sbml_level = 2
sbml_version = 4

def rxn_add_stoich(srxn, rid, stoich, is_product=True):

    try:
        stoich = float(stoich)
        if stoich < 0:
            sr = srxn.createReactant()
            sr.setStoichiometry(-stoich)
        elif stoich > 0:
            sr = srxn.createProduct()
            sr.setStoichiometry(stoich)
        elif stoich == 0:
            sr = srxn.createModifier()
        sr.setSpecies(rid)

    except ValueError:
        formula = stoich.replace('**', '^')
        math_ast = libsbml.parseFormula(formula)
        #try:
        #    smath = libsbml.StoichiometryMath(math_ast)
        #except:
        # normally this is in an except block but the try above doesn't throw erro
        # I am expecting in libSBML 4.0
        try:
            smath = libsbml.StoichiometryMath(math_ast)
        except NotImplementedError:
            smath = libsbml.StoichiometryMath(sbml_level, sbml_version)
            smath.setMath(math_ast)

        if is_product == True:
            sr = srxn.createProduct()
        else:
            sr = srxn.createReactant()

        sr.setSpecies(rid)
        sr.setStoichiometryMath(smath)

def fromSBMLFile(fileName, id = None, duplicate_rxn_params=False):
    f = file(fileName, 'r')
    net = fromSBMLString(f.read(), id, duplicate_rxn_params)
    f.close()
    return net

def stoichToString(species, stoich):
    if stoich is None:
        stoich = str(species.getStoichiometry())
    elif hasattr(stoich, 'getMath'): # libsbml > 3.0
        stoich = libsbml.formulaToString(stoich.getMath())
    else: # libsbml 2.3.4
        stoich = libsbml.formulaToString(stoich) 
    return stoich    


class SBMLNotesParser(HTMLParser.HTMLParser):
    """ Extract key-value pairs from HTML-formatted SBML Notes elements. 

    Data which does not appear to contain a key: value pair will be collected
    and stored under the key None. 

    """
    def __init__(self): 
        HTMLParser.HTMLParser.__init__(self)
        self.records = {}
        self.other_data = []
    def handle_data(self, data):
        if not data.isspace():
            if ':' in data:
                key, value = data.split(':', 1)
                self.records[key.strip()] = value.strip()
            else:
                self.other_data.append(data)

def load_notes(libsbml_component, network_component):
    if libsbml_component.isSetNotes():
        parser = SBMLNotesParser()
        parser.feed(libsbml_component.getNotesString())
        result = parser.records
        if parser.other_data:
            result[None] = parser.other_data
        network_component.notes = result
    else:
        network_component.notes = {}

def write_sbml_note_table(dictionary):
    """Convert key-value dictionary to COBRA-style SBML notes.

    Returns an HTML string suitable for use as the optional
    'notes' element on an SBML component. This may be verified with
    libsbml.SyntaxChecker_hasExpectedXHTMLSyntax(). 

    The value associated with the key None, if any, will be 
    appended at the end of the notes string, rather than 
    formatted.

    Per the supplementary information to doi:10.1038/nprot.2011.308

    """
    
    records = []
    for key,value in dictionary.iteritems():
        if key is None:
            continue
        records.append('\n    <p>%s: %s</p>' % (key, value))
    if None in dictionary:
        # If the notes were written with load_notes above, the None
        # record should be a list of miscellaneous not parsed as
        # key-value pairs. We somewhat awkwardly dump this all
        # together at the end of the notes field.
        records.append('\n' + '\n'.join(dictionary[None]))

    template = "<body xmlns='http://www.w3.org/1999/xhtml'>%s\n</body>"
    return template % ''.join(records)

def save_notes(network_component, libsbml_component):
    if hasattr(network_component, 'notes') and network_component.notes:
        libsbml_component.setNotes(write_sbml_note_table(network_component.notes))

def fromSBMLString(sbmlStr, id = None, duplicate_rxn_params=False):
    """Load a reaction network from SBML.
    
    Function definitions, events, and constraints are ignored, as our
    modeling framework cannot handle them.

    Notes attributes are read for comparmtents, reactions, species,
    and parameters, as well as the document itself, but not for rules
    (those are stored only as math expression strings; there is
    nowhere to collect additional annotations for them.

    """
    r = libsbml.SBMLReader()
    d = r.readSBMLFromString(sbmlStr)
    # Here the old version would have checked for errors. Use this fn with 
    # potentially invalid SBML at your own risk!

    m = d.getModel()

    modelId = m.getId()
    if (id == None) and (modelId == ''):
        raise ValueError('Network id not specified in SBML or passed in.')
    elif id is not None:
        modelId = id
        
    rn = Network_mod.Network(id = modelId, name = m.getName())
    load_notes(m, rn)

    for c in m.getListOfCompartments():
        id, name = c.getId(), c.getName()
        size = c.getSize()
        isConstant = c.getConstant()

        rn.add_compartment(id = id, initial_size = size, 
                          is_constant = isConstant, 
                          name = name)
        load_notes(c, rn.compartments.getByKey(id))

    for s in m.getListOfSpecies():
        id, name = s.getId(), s.getName()
        compartment = s.getCompartment()
        if s.isSetInitialConcentration():
            iC = s.getInitialConcentration()
        elif s.isSetInitialAmount():
            iC = s.getInitialAmount()
        else:
            iC = 1
        isBC, isConstant = s.getBoundaryCondition(), s.getConstant()

        xml_text = s.toSBML()
        uniprot_ids = set([entry[1:].split('"')[0] 
                           for entry in xml_text.split('uniprot')[1:]])
	
	rn.add_species(id = id, compartment = compartment,
                      initial_conc = iC,
                      is_constant = isConstant,
                      is_boundary_condition = isBC,
                      name = name, uniprot_ids = uniprot_ids)
        load_notes(s, rn.species.getByKey(id))

    for p in m.getListOfParameters():
        parameter = createNetworkParameter(p)
        rn.add_parameter(parameter)
        load_notes(p, rn.parameters.getByKey(p))

    for rxn in m.getListOfReactions():
        id, name = rxn.getId(), rxn.getName()

        # If the kinetic law is set, get it and extract
        # any parameters that may be encoded in it, as COBRA
        # flux bounds and objective parameters.
        # TO DO: test that this does not lead to problems
        # if kinetic laws are present and empty (in violation of
        # the standard)
        if rxn.isSetKineticLaw():
            kinetic_law = rxn.getKineticLaw()
            kinetic_law_parameters = {parameter.getId(): parameter.getValue()
                                      for parameter in
                                      kinetic_law.getListOfParameters()}
            kinetic_law_math = kinetic_law.getFormula()
        else:
            kinetic_law_parameters = {}
            kinetic_law_math = None
    
        # Assemble the stoichiometry. SBML has the annoying trait that 
        #  species can appear as both products and reactants and 'cancel out'
        # For each species appearing in the reaction, we build up a string
        # representing the stoichiometry. Then we'll simplify that string and
        # see whether we ended up with a float value in the end.
        stoichiometry = {}
        reactant_stoichiometry = {}
        product_stoichiometry = {}
        for reactant in rxn.getListOfReactants():
            species = reactant.getSpecies()
            stoichiometry.setdefault(species, '0')
            stoich = reactant.getStoichiometryMath()
            stoich = stoichToString(reactant, stoich)
            stoichiometry[species] += '-(%s)' % stoich
            if species in reactant_stoichiometry:
                reactant_stoichiometry[species].append(stoich)
            else:
                reactant_stoichiometry[species] = [stoich]
    
        for product in rxn.getListOfProducts():
            species = product.getSpecies()
            stoichiometry.setdefault(species, '0')
            stoich = product.getStoichiometryMath()
            stoich = stoichToString(product, stoich)
            stoichiometry[species] += '+(%s)' % stoich
            if species in product_stoichiometry:
                product_stoichiometry[species].append(stoich)
            else:
                product_stoichiometry[species] = [stoich]

        for species, stoich in stoichiometry.items():
            stoich = expr_manip.simplify_expr(stoich)
            try:
                # Try converting the string to a float.
                stoich = float(stoich)
            except ValueError:
                pass
            stoichiometry[species] = stoich

        for modifier in rxn.getListOfModifiers():
            stoichiometry.setdefault(modifier.getSpecies(), 0)

        rn.addReaction(id = id, stoichiometry = stoichiometry,
                       kineticLaw = kinetic_law_math,
                       reactant_stoichiometry = reactant_stoichiometry,
                       product_stoichiometry = product_stoichiometry,
                       name = name, parameters = kinetic_law_parameters)
        load_notes(rxn, rn.reactions.getByKey(id))
        rn.reactions.getByKey(id).reversible = rxn.getReversible()

    for ii, r in enumerate(m.getListOfRules()):
        if r.getTypeCode() == libsbml.SBML_ALGEBRAIC_RULE:
            math = libsbml.formulaToString(r.getMath())
            rn.add_algebraic_rule(math)
        else:
            variable = r.getVariable()
            math = libsbml.formulaToString(r.getMath())
            if r.getTypeCode() == libsbml.SBML_ASSIGNMENT_RULE:
                rn.add_assignment_rule(variable, math)
            elif r.getTypeCode() == libsbml.SBML_RATE_RULE:
                rn.add_rate_rule(variable, math)
    return rn

#### file break was here

def toSBMLString(net):
    try:
        m = libsbml.Model(net.id)
    except NotImplementedError:
        m = libsbml.Model(sbml_level, sbml_version)
        m.setId(net.id)
    m.setName(net.name)
    
    for id, c in net.compartments.items():
        try:
            sc = libsbml.Compartment(id)
        except NotImplementedError:
            sc = libsbml.Compartment(sbml_level, sbml_version)
            sc.setId(id)
        sc.setName(c.name)
        sc.setConstant(c.is_constant)
        sc.setSize(c.initialValue)
        save_notes(c, sc)
        m.addCompartment(sc)
    
    for id, s in net.species.items():
        try:
            ss = libsbml.Species(id)
        except NotImplementedError:
            ss = libsbml.Species(sbml_level, sbml_version)
            ss.setId(id)
        ss.setName(s.name)
        ss.setCompartment(s.compartment)
        if s.initialValue is not None and not isinstance(s.initialValue, str):
            ss.setInitialConcentration(s.initialValue)
        ss.setBoundaryCondition(s.is_boundary_condition)
        save_notes(s, ss)
        m.addSpecies(ss)
    
    for id, p in net.parameters.items():
        try:
            sp = libsbml.Parameter(id)
        except NotImplementedError:
            sp = libsbml.Parameter(sbml_level, sbml_version)
            sp.setId(id)
        sp.setName(p.name)
        if p.initialValue is not None:
            sp.setValue(p.initialValue)
        sp.setConstant(p.is_constant)
        save_notes(p, sp)
        m.addParameter(sp)

    for id, r in net.rateRules.items():
        try:
            sr = libsbml.RateRule()
        except NotImplementedError:
            sr = libsbml.RateRule(sbml_level, sbml_version)
        sr.setVariable(id)
        formula = r.replace('**', '^')
        sr.setMath(libsbml.parseFormula(formula))
        m.addRule(sr)

    for id, r in net.assignmentRules.items():
        try:
            sr = libsbml.AssignmentRule()
        except NotImplementedError:
            sr = libsbml.AssignmentRule(sbml_level, sbml_version)
        sr.setVariable(id)
        formula = r.replace('**', '^')
        sr.setMath(libsbml.parseFormula(formula))
        m.addRule(sr)

    for r, r in net.algebraicRules.items():
        try:
            sr = libsbml.AlgebraicRule()
        except NotImplementedError:
            sr = libsbml.AlgebraicRule(sbml_level, sbml_version)
        formula = r.replace('**', '^')
        sr.setMath(libsbml.parseFormula(formula))
        m.addRule(sr)
        
    for id, rxn in net.reactions.items():
        try:
            srxn = libsbml.Reaction(id)
        except NotImplementedError:
            srxn = libsbml.Reaction(sbml_level, sbml_version)
            srxn.setId(id)
        srxn.setName(rxn.name)
        save_notes(rxn, srxn)
        # Handle the case where the model was originally read in from an
        # SBML file, so that the reactants and products of the Reaction
        # object are explicitly set.
        if rxn.reactant_stoichiometry != None and \
            rxn.product_stoichiometry != None:
            for rid, stoich_list in rxn.reactant_stoichiometry.items():
                for stoich in stoich_list:
                    # 'stoich' is a string. If it may be easily cast
                    # to a float, we expect it to be a positive float;
                    # however, if we pass either a float or a string
                    # which may be converted to a float to
                    # rxn_add_stoich, the sign of that float will
                    # determine whether rxn_add_stoich adds a reactant
                    # or a product, so we may need to multiply by -1.0
                    # here. It is also possible that 'stoich' is a
                    # math expression, in which case the 'is_product'
                    # argument controls whether a reactant or product
                    # is added. 
                    try:
                        stoich = float(stoich)
                        rxn_add_stoich(srxn, rid, -stoich, is_product=False)
                    except ValueError:    
                        rxn_add_stoich(srxn, rid, stoich, is_product=False)
            for rid, stoich_list in rxn.product_stoichiometry.items():
                for stoich in stoich_list:
                    rxn_add_stoich(srxn, rid, stoich, is_product=True)
        # Handle the case where the model was created using the SloppyCell
        # API, in which case reactants and products are inferred from their
        # stoichiometries
        else:
            for rid, stoich in rxn.stoichiometry.items():
                rxn_add_stoich(srxn, rid, stoich)

        # Ensure kinetic laws exist (as they may not for FBA-type models)
        # -- not clear what a good default kinetic law is; use '0',
        # which should at least lead to unsubtle problems
        if not rxn.kineticLaw:
            formula = '0'
        else:
            formula = rxn.kineticLaw
        formula = formula.replace('**', '^')
        try:
            kl = libsbml.KineticLaw(formula)
        except NotImplementedError:
            kl = libsbml.KineticLaw(sbml_level, sbml_version)
            kl.setFormula(formula)
        srxn.setKineticLaw(kl)

        # Set the optional reversibility attribute, if one is present.
        if hasattr(rxn,'reversible'):
            srxn.setReversible(rxn.reversible)
        m.addReaction(srxn)
    
        # Set the kinetic law's parameters
        # Note that, properly, we should be handling units and other
        # attributes here; we are most concerned about COBRA SBML
        # models, where these appear to be of secondary importance
        # (Yeastnet 7, for example, defines a lot of these 
        # parameters as 'dimensionless'.) 
        for parameter, value in getattr(rxn, 'parameters', {}).iteritems():
            # The createKineticLawParameter method creates a parameter
            # in the most kinetic law of the most recently created
            # reaction in the model.

            sparameter = m.createKineticLawParameter()
            print sparameter
            sparameter.setId(parameter)
            sparameter.setValue(value)

    d = libsbml.SBMLDocument(sbml_level, sbml_version)
    d.setModel(m)
    save_notes(net, d)
    sbmlStr = libsbml.writeSBMLToString(d)

    return sbmlStr

def toSBMLFile(net, fileName):
    sbmlStr = toSBMLString(net)
    f = file(fileName, 'w')
    f.write(sbmlStr)
    f.close()

def createNetworkParameter(p):
    id, name = p.getId(), p.getName()
    v = p.getValue()
    isConstant = p.getConstant()
    parameter = Network_mod.Parameter(id = id, value = v, is_constant
                                      = isConstant, name = name,
                                      typical_value = None,
                                      is_optimizable = True)
                                  # optimizable by default
    return parameter
