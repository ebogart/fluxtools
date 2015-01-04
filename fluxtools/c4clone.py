"""
Utilities for turning a one-cell SloppyCell network into a two-cell network.

"""

from reaction_networks import *

def strip_empty_compartments(network):
    """ Remove compartments from network if they contain no species.

    """ 
    occupied_compartment_ids = set([])
    for s in network.species:
        occupied_compartment_ids.add(s.compartment)
    
    print occupied_compartment_ids

    for compartment_id in network.compartments.keys():
        if compartment_id not in occupied_compartment_ids:
            network.compartments.removeByKey(compartment_id)

def clone(source, join_compartment_ids, new_network_id, new_name, a_label='ms',
          b_label='bs', reversibility_list = []):
    """
    Make a two-cell network from a one-cell network.

    Returns a new network and a dictionary mapping ids in the old
    network to ids or pairs of ids in the new network as appropriate.

    If reversibility_list is given, a list of the reactions in the new model
    corresponding to the entries of reversibility_list is returned as well.

    Note that the duplication of network components is not perfect:
    only compartments, species and reactions are considered, and attributes
    of those objects beyond what is relevant to FBA are currently ignored
    (initial values, kinetic laws, eg.) Note further that the component
    ids in the cloned network are not guaranteed to be valid SBML ids, though
    if a_label and b_label are valid beginnings of SBML ids and the old network
    ids are valid, they should be.

    """
    n = Network(new_network_id, new_name)
    
    a_tag = lambda s: a_label + '_' + s
    b_tag = lambda s: b_label + '_' + s
    
    replacement_table = {}

    # We will remove old reactions from the list of reversible reactions
    # one at a time, so we want to ensure each reaction appears at most once
    reversibility_list = list(set(reversibility_list))

    for c in source.compartments:
        if c.id in join_compartment_ids:
            n.addCompartment(c.id,name = c.name)
            replacement_table[c.id] = c.id
        else:
            n.addCompartment(a_tag(c.id),name = a_tag(c.name))
            n.addCompartment(b_tag(c.id),name = b_tag(c.name))
            replacement_table[c.id] = (a_tag(c.id), b_tag(c.id))

    for s in source.species:
        if s.compartment in join_compartment_ids:
            n.addSpecies(s.id,s.compartment,name = s.name)
            replacement_table[s.id] = s.id
        else:
            n.addSpecies(a_tag(s.id),a_tag(s.compartment),
                                      name = a_tag(s.name)) 
            n.addSpecies(b_tag(s.id),b_tag(s.compartment),
                                      name = b_tag(s.name)) 
            replacement_table[s.id] = (a_tag(s.id), b_tag(s.id)) 
            
    for r in source.reactions:
        species = [source.species.getByKey(s) for s in r.stoichiometry.keys()]
        if all([s.compartment in join_compartment_ids for s in species]):
            n.addReaction(Reactions.Reaction, r.id, r.stoichiometry, 
                                             name=r.name, kineticLaw = None)
            replacement_table[r.id] = r.id
        else:
            if r.id in reversibility_list:
                reversibility_list.remove(r.id)
                reversibility_list += [a_tag(r.id), b_tag(r.id)]
            a_stoichiometry = {}
            b_stoichiometry = {}
            for s in species:
                if s.compartment in join_compartment_ids:
                    a_stoichiometry[s.id] = r.stoichiometry[s.id]
                    b_stoichiometry[s.id] = r.stoichiometry[s.id]
                else:
                    a_stoichiometry[a_tag(s.id)] = r.stoichiometry[s.id]
                    b_stoichiometry[b_tag(s.id)] = r.stoichiometry[s.id]

            n.addReaction(Reactions.Reaction,a_tag(r.id), a_stoichiometry, 
                                             name=a_tag(r.name),
                                             kineticLaw = None)
            n.addReaction(Reactions.Reaction,b_tag(r.id), b_stoichiometry, 
                                             name=b_tag(r.name),
                                             kineticLaw = None)

            replacement_table[r.id] = (a_tag(r.id), b_tag(r.id))

    if reversibility_list:
        return n, replacement_table, reversibility_list
    else: 
        return n, replacement_table
    
def give_exchange_reaction(network, species_pair, prefix='plasmodesmata_',
                           name = None):
    """ Add a reaction to the network that interconverts two species. """
    r_id = prefix + species_pair[0] + '_' + species_pair[1]
    if name is None:
        name = r_id
    network.addReaction(Reactions.Reaction, r_id,
                        dict(zip(species_pair, (1.0, -1.0))),
                        name = name,
                        kineticLaw = None)
    return r_id
