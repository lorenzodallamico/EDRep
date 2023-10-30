import numpy as np
from node2vec.model import Node2Vec

from EDRep import *


def NodeEmbedding(A, dim, n_epochs = 30, walk_length = 5, k = 1, verbose = True, η = 0.5, sym = True):
    '''Algorithm for node embedding using Eder
    
    * Use: res = NodeEmbedding(A, dim)

    * Inputs:
        * A (scipy sparse matrix): graph adjacency matrix. It can be weighted and non-symmetric, but its entries must be non-negative
        * dim (int): embedding dimension
        
    * Optional inputs:
        * n_epochs (int): number of training epochs in the optimization. By default set to 30
        * walk_length (int): maximal distance reached by the random walker. By default set to 5
        * k (int): order of the mixture of Gaussian approximation. By default set to 1
        * verbose (bool): if True (default) it prints the update
        * η (float): learning rate, by default set to 0.5
        * sym (bool): determines whether to use the symmetric (detfault) version of the algoritm
        
    * Output:
        * res: EDREp class
    '''
    
    # check that all entries of A are positive
    if not (A[A.nonzero()] > 0).all():
        raise DeprecationWarning("The weighted adjacency matrix has negative entries")
    
    # create the probability matrix
    n = A.shape[0]
    d = A@np.ones(n)
    D_1 = diags(d**(-1))
    P = D_1.dot(A)
 
    # Eder
    embedding = CreateEmbedding([P for i in range(walk_length)], dim = dim, n_epochs = n_epochs, 
                        sum_partials = True, k = k, verbose = verbose, η = η, sym = sym)
    
    return embedding


def Node2VecNS(A, dim, verbose):
    '''This function compute the Node2Vec embedding with negative sampling, using the standard function parameters
    
    Use: X = Node2Vec(A, dim, verbose)
    
    Input: 
        * A (sparse csr_matrix): sparse adjacency matrix of the graph
        * dim (int): embedding dimensionality
        * verbose (bool): sets the level of verbosity
        
    Output:
        * X (array): embedding matrix
        
    '''

    src_nodes, dest_nodes = A.nonzero()
    node2vec_model = Node2Vec(src_nodes, dest_nodes, graph_is_directed = False)
    node2vec_model.simulate_walks(workers = 8, verbose = verbose, p = 1, q = 1)
    node2vec_model.learn_embeddings(dimensions = dim, workers = 8, verbose = verbose)
    X = node2vec_model.embeddings

    return X

