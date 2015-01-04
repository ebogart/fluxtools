""" Tools for working with reaction-gene associations in models.

Currently only 'flat' assocations of the form 
'GENE_ID_1 [or GENE_ID_2 [or GENE_ID_3 ... ]]' are supported.  

"""

gra_field = 'GENE_ASSOCIATION'

def to_GRA_string(list_of_genes):
    """Form a COBRA-style GRA rule indicating an 'or' relationship.

    >>> to_GRA_string(['GRMZM2G000001', 'GRMZM2G000002'])
    'GRMZM2G000001 or GRMZM2G000002'
   
    """
 
    return ' or '.join(list_of_genes)

def set_gene_association_from_list(reaction, list_of_genes):
    """Add a COBRA-style GRA rule association the reaction with the genes.
    
    The rule will be 'gene1 or gene2 or ...'

    Any existing rule will be overwritten.

    """
    reaction.notes[gra_field] = to_GRA_string(list_of_genes)

def genes_of_reaction(reaction):
    genes = reaction.notes.get(gra_field,None)
    if genes:
        return genes.split(' or ')
    else:
        return []

def get_gra(net):
    """ Extract GRA from 'GENE_ASSOCIATION' notes.

    Currently this is very fragile and can handle only ' or ' 
    relationships without parentheses, etc.

    """
    gra = {}
    for reaction in net.reactions:
        genes = genes_of_reaction(reaction)
        for g in genes:
            gra.setdefault(reaction.id,set()).add(g)
    return gra

