import sets
from .. import expr_manip

class Reaction:
    def __init__(self, id, stoichiometry, kineticLaw = '', name = '',
                 reactant_stoichiometry = None, product_stoichiometry = None,
                 parameters={}):
        self.id = id
        self.stoichiometry = stoichiometry
        # self.reactant_stoichiometry and self.product_stoichiometry
        # are defined to help repserve the stoichiometry defined in an
        # SBML model
        self.reactant_stoichiometry = reactant_stoichiometry
        self.product_stoichiometry = product_stoichiometry
        self.kineticLaw = kineticLaw
        self.name = name
        self.parameters = parameters

    def __eq__(self, other):
        return self.__class__ == other.__class__ and \
                self.kineticLaw == other.kineticLaw and \
                self.stoichiometry == other.stoichiometry

    def __ne__(self, other):
        return not (self == other)

    def doKwargsSubstitution(self, kwargs):
        oldStoichiometry = self.stoichiometry
        self.stoichiometry = {}
        for base in oldStoichiometry:
            self.stoichiometry[kwargs[base]] = oldStoichiometry[base]

        self.kineticLaw = expr_manip.sub_for_vars(self.kineticLaw, kwargs)

    def change_stoichiometry(self, species, stoich):
        """
        Change stoichiometry will update self.stoichiometry.
        It also updates self.reactant_stoichiometry and self.product_stoichiometry,
        if they are defined, to keep the two stoichiometry tracking systems in sync.
        """
        self.stoichiometry[species] = stoich
        if self.reactant_stoichiometry != None and self.product_stoichiometry != None:
            if species in self.reactant_stoichiometry.keys():
                del self.reactant_stoichiometry[species]
            self.product_stoichiometry[species] = [stoich]

