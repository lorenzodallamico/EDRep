import numpy as np
from scipy.sparse import diags, csr_matrix

from EDRep import CreateEmbedding


def NodeEmbedding(A: csr_matrix, dim: int, n_epochs: int = 30, walk_length: int = 5, k: int = 1, verbose: bool = True, η: float = 0.5, sym: bool = True):
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
        * sym (bool): determines whether to use the symmetric (default) version of the algoritm
        
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
 
    # EDRep
    embedding = CreateEmbedding([P for i in range(walk_length)], dim = dim, n_epochs = n_epochs, 
                        sum_partials = True, k = k, verbose = verbose, η = η, sym = sym)
    
    return embedding



