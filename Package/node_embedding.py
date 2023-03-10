import numpy as np
from node2vec.model import Node2Vec

from eder import *


_flat = lambda x:x**0


def NodeEmbedding(A, dim, f_func = _flat, n_epochs = 30, n_prod = 1, k = 1, cov_type = 'full', verbose = True, η = 0.85):
    '''Algorithm for node embedding using Eder
    
    * Use: X = NodeEmbedding(A, dim, f, n_epochs = 35, n_prod = 1, k = 1, cov_type = 'diag', verbose = True, η0 = 0.8)
    
    * Inputs:
        * A (scipy sparse matrix): graph adjacency matrix. It can be weighted and non-symmetric, but its entries must be non-negative
        * dim (int): embedding dimension
        
    * Optional inputs:
        * f_func (function): the norm of x_i will be set to f(d_i)
        * n_epochs (int): number of training epochs in the optimization. By default set to 35   
        * n_prod (int): maximal distance reached by the random walker. By default set to 1
        * k (int): order of the mixture of Gaussian approximation. By default set to 1
        * cov_type (string): determines the covariance type in the optimization process. Can be 'diag' or 'full'
        * verbose (bool): if True (default) it prints the update
        * η (float): learning rate
        
    * Output:
        * X (array): embedding matrix
    '''
    
    # check that all entries of A are positive
    if not (A[A.nonzero()] > 0).all():
        raise DeprecationWarning("The weighted adjacency matrix has negative entries")
    
    # create the probability matrix
    n = A.shape[0]
    d = A@np.ones(n)
    D_1 = diags(d**(-1))
    P = D_1.dot(A)

    # bound the distribution to its 95 percentile
    f = f_func(d)

    # apply a threshold for the decay
    th = np.sort(f)[int(0.95*n)]
    f[f > th] = th*np.sqrt(np.log(f[f > th])/np.log(th))   

    # normalize
    f = f/np.mean(f)    
    
    # Eder
    X = CreateEmbedding([P], f = f, dim = dim, n_epochs = n_epochs, n_prod = n_prod, sum_partials = True,
                      k = k, verbose = verbose, cov_type = cov_type, η = η)
    
    return X


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

