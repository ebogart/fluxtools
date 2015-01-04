"""Identify blocked reactions/unused species, remove from a network.

The Graph classes are adapted from Cornell Phys. 7682 exercise, fall
2009; most of the code comes from the template for the Networks
problem.

"""
import logging
import numpy as np

class UndirectedGraph:
    """An UndirectedGraph g contains a dictionary (g.connections) that
    maps a node identifier (key) to a list of nodes connected to
    (values).  g.connections[node] returns a list [node2, node3,
    node4] of neighbors.  Node identifiers can be any non-mutable
    Python type (e.g., integers, tuples, strings, but not lists).

    """

    def __init__(self):
        """UndirectedGraph() creates an empty graph g.
	g.connections starts as an empty dictionary.  When nodes are
	added, the corresponding values need to be inserted into lists.

        A method/function definition in a class must begin with an
        instance of the class in question; by convention, the name
        "self" is used for this instance.

        """
        self.connections = {}

    def HasNode(self, node):
        """Returns True if the graph contains the specified node, and False
        otherwise.  Check directly to see if the dictionary of
        connections contains the node, rather than (inefficiently)
        generating a list of all nodes and then searching for the
        specified node.

        """
        # return self.connections.has_key(node)
        # or instead, using the "in" operator that calls has_key():
        if node in self.connections:
            return True
        else:
            return False

    def AddNode(self, node):
	"""Uses HasNode(node) to determine if node has already been added."""
        if self.HasNode(node):
            pass
        else:
            self.connections[node] = [];

    def AddEdge(self, node1, node2):
        """
	Add node1 and node2 to network first
	Adds new edge 
	(appends node2 to connections[node1] and vice-versa, since it's
	an undirected graph)
	Do so only if old edge does not already exist 
	(node2 not in connections[node1])
	"""
        self.AddNode(node1);
        self.AddNode(node2);

        if node2 not in self.connections[node1]:
            self.connections[node1].append(node2)
        if node1 not in self.connections[node2]: # Redundant, but I'm paranoid
            self.connections[node2].append(node1)
        

    def GetNodes(self):
        """g.GetNodes() returns all nodes (keys) in connections"""
        return self.connections.keys();

    def GetNeighbors(self, node):
        """g.GetNeighbors(node) returns a copy of the list of neighbors of the
        specified node.  A copy is returned (using the [:] operator)
        so that the user does not inadvertently change the neighbor
        list.

        """
        return self.connections[node][:] # think this is the indicated operator


class ReactionSpeciesGraph (UndirectedGraph):
    """An undirected bipartite reaction-species graph with some useful
    utilities. Note that the ancestral add_node, add_edge methods
    still work, but will generate nodes which are neither reactions
    nor species, breaking some methods.

    """

    def __init__(self):
        """UndirectedGraph() creates an empty graph g.
	g.connections starts as an empty dictionary.  When nodes are
	added, the corresponding values need to be inserted into lists.

        A method/function definition in a class must begin with an
        instance of the class in question; by convention, the name
        "self" is used for this instance.

        """
        self.connections = {}
        self.reactions = set([])
        self.species = set([])

    def AddSpecies(self, species):
        self.AddNode(species)
        self.species.add(species)

    def AddReaction(self, reaction, stoichiometry):
        self.AddNode(reaction)
        self.reactions.add(reaction)
        for s in stoichiometry:
            if stoichiometry[s] != 0.0:
                self.species.add(s)
                self.AddEdge(reaction, s)

    def ExtirpateSpecies(self, species):
        """Remove a species and all reactions involving it from the
        network.

        """
        # We need to iterate through the reactions involving this
        # species and remove them from the network, but that process
        # modifies the species' connection list, so we must iterate
        # over a copy of the original version of the list:
        involving_reactions = self.connections[species][:] 
        for r in involving_reactions:
            self.RemoveReaction(r)
        self.connections.pop(species)
        self.species.remove(species)

    def RemoveReaction(self, reaction):
        reactants = self.connections.pop(reaction)
        for s in reactants:
            self.connections[s].remove(reaction)
        self.reactions.remove(reaction)

    def OrphanSpecies(self):
        return [s for s in self.species if len(self.connections[s]) < 2]

    def BlockedReactions(self):
        blocked = []
        for orphan in self.OrphanSpecies():
            blocked += self.connections[orphan]
        # However, a species may involve multiple blocked reactions 
        # and be listed more than once, so we remove duplicates.
        return list(set(blocked))

    def DestroyOrphans(self):
        """Extirpate the orphan species, removing blocked reactions from the
        model.

        """
        orphans = self.OrphanSpecies()
        num_reactions = len(self.reactions)
        for s in orphans:
            self.ExtirpateSpecies(s)
        
def deblock(stoichiometry_dictionary, protect_species=()):
    """Remove blocked reactions and singleton metabolites.

    These are removed from a reaction network (provided as a
    stoichiometry dictionary of the form {'reaction_id': {'product1',
    1.0, 'reactant1', -1.0, ...}, ...}), repeating until none remain.
    
    Species in the protect_species argument are never considered
    singleton metabolites, do not block reactions in which they
    participate, and are not removed from the model even if all reactions
    involvign them are removed.

    A verbose report on the process is written to logging.INFO. 

    Where species which have been removed from the network participate
    in reactions remaining in the network with stoichiometric
    coefficient zero, they will be removed from the stoichiometries of
    those reactions.

    """
    graph = ReactionSpeciesGraph()

    for reaction in stoichiometry_dictionary:
        graph.AddReaction(reaction, stoichiometry_dictionary[reaction])

    logging.info(str(len(stoichiometry_dictionary)) + ' reactions in dictonary.\n')
    n_reactions = len(graph.reactions)
    n_species = len(graph.species)
    initial_cluster_structure = [len(c) for c in FindAllClusters(graph)]
    n_clusters = len(initial_cluster_structure)

    logging.info(str(n_reactions) + ' reactions and ' + str(n_species) + 
            ' species in initial graph, in ' + str(n_clusters) + 
            ' cluster(s) of size(s):\n' + str(initial_cluster_structure)
            + '\n')

    step = 0 
    singletons = [s for s in graph.OrphanSpecies() if s not in protect_species]
    true_singletons = singletons[:]
    removed_species = []
    removed_reactions = []

    recently_blocked_by_reactant = {}

    while singletons:
        step += 1
        logging.info('\nRound ' + str(step) + ' of pruning:\n') 

        blocked_reactions = set()

        for species in singletons:
            if graph.connections[species]: # a list of length at most 1
                blocked_reaction = graph.connections[species][0]
                blocked_reactions.add(blocked_reaction)
                logging.info(str(species) + ' participates only in ' + 
                        str(blocked_reaction))
                if species in recently_blocked_by_reactant:
                    logging.info(' (previously also participated in ' +
                            str(recently_blocked_by_reactant[species]) +
                            ')')
                logging.info('.\n')
            else:
                logging.info(str(species) + 
                        ' participates in no remaining reactions')
                if species in recently_blocked_by_reactant:
                    logging.info(' (previously also participated in ' +
                            str(recently_blocked_by_reactant[species]) +
                            ')')
                logging.info('.\n')

        recently_blocked_by_reactant = {}
        for r in blocked_reactions:
            for s in graph.connections[r]:
                if s not in singletons:
                    if s in recently_blocked_by_reactant:
                        recently_blocked_by_reactant[s].append(r)
                    else:
                        recently_blocked_by_reactant[s] = [r]
                
        for species in singletons:
            graph.ExtirpateSpecies(species)

        removed_species += singletons
        removed_reactions += list(blocked_reactions)

        singletons = [s for s in graph.OrphanSpecies() if s not in protect_species]

    # Final size and structure of output
    n_species_removed = len(removed_species)
    n_reactions_removed = len(removed_reactions)

    logging.info('\nPruning complete after ' + str(step) + ' rounds; ' +
            str(n_reactions_removed) + ' reactions and ' + 
            str(n_species_removed) + ' species removed.\n')

    n_reactions = len(graph.reactions)
    n_species = len(graph.species)
    final_cluster_structure = [len(c) for c in FindAllClusters(graph)]
    n_clusters = len(final_cluster_structure)

    logging.info(str(n_reactions) + ' reactions and ' + str(n_species) + 
            ' species in final graph, in ' + str(n_clusters) + 
            ' cluster(s) of size(s):\n' + str(final_cluster_structure)
            + '\n') 

    new_stoichiometries =  dict([(reaction, 
                                  stoichiometry_dictionary[reaction]) for 
                                 reaction in graph.reactions])

    for r in new_stoichiometries:
        for s, c in new_stoichiometries[r].items():
            if s in removed_species:
                if c == 0.0:
                    new_stoichiometries[r].pop(s)
                else:
                    raise Exception('Singleton metabolite appears '
                                    'to participate in non-blocked species.')

    return new_stoichiometries, removed_reactions, true_singletons


def FindClusterFromNode(graph, node, visited=None):
    """Breadth--first search

    The dictionary "visited" should be initialized to False for
    all the nodes in the cluster you wish to find
    It's used in two different ways.
    (1) It's passed back to the
        calling program with all the nodes in the current cluster set to
        visited[nodeInCluster]=True, so that the calling program can skip
        nodes in this cluster in searching for other clusters.
    (2) It's used internally in this algorithm to keep track of the
        sites in the cluster that have already been found and incorporated
    See "Building a Percolation Network" in text for algorithm

    """
    visited[node] = True;
    cluster = [node];
    currentShell = graph.GetNeighbors(node)
    #print currentShell
    while len(currentShell) != 0:
        #print 'in loop'
        nextShell = [];
        for currentNode in currentShell:
            #print currentNode
            if not visited[currentNode]:
                visited[currentNode] = True
                cluster.append(currentNode)
                nextShell = nextShell + graph.GetNeighbors(currentNode)
        currentShell = nextShell;
        
    return cluster

def FindAllClusters(graph):
    """For example, find percolation clusters
    Set up the dictionary "visited" for FindClusterFromNode
    Set up an empty list "clusters"
    Iterate over the nodes;
        if it haven't been visited,
            find the cluster containing it
            append it to the cluster list
        return clusters
    Check your answer using
    NetGraphics.DrawSquareNetworkBonds(g, cl) and
    NetGraphics.DrawSquareNetworkSites(g, cl)
					            
    Optional: You may wish to sort your list of clusters according to their
    lengths, biggest to smallest
    For a list ell, the built-in method ell.sort() will sort the list
    from smallest to biggest;
    ell.sort(cmp) will sort the list according to the comparison function
    cmp(x, y) returns -1 if x < y, returns 0 if x==y, and returns 1 if x>y
    Define ReverseLengthCompare to compare two lists according to the
    unusual definition of inequality, l1<l2 if # len(l1) > len(l2)!

    """
    nodes = graph.GetNodes();
    clusters = [];
    visited = dict.fromkeys(nodes,False);

    for node in nodes:
        if not visited[node]:
            clusters.append(FindClusterFromNode(graph, node, visited))
    return clusters
    clusters.sort(ReverseLengthCompare)

def ReverseLengthCompare(a,b):
    """compare two lists according to the
    unusual definition of inequality, l1<l2 if # len(l1) > len(l2)!
    """
    if len(a) > len (b):
        return -1
    elif len(b) > len(a):
        return 1
    else:
        return 0

def GetSizeDistribution(clusters,nBins=25):
    """ Return a list of two arrays, one of the low end of the exponential bins
    and one of the number of clusters within that bin.

    """

    logSizes = [np.log(len(cluster)) for cluster in clusters]
    bins = scipy.linspace(0,max(logSizes),nBins);
    distribution = zeros(nBins)
    for i in range (nBins-1):
        sizesInBin = [size for size in logSizes if (size>bins[i] and size<bins[i+1])]
        distribution[i] = len(sizesInBin)

    distribution[nBins-1] = len([size for size in logSizes if size>bins[nBins-1]])

    return [bins,distributions]

    
    
        

