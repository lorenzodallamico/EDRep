import numpy as np
from scipy.sparse import diags

def RFM(X, D, indeces):
    '''Softmax approximation using RFM algorithm
    
    Use: Z = RFM(X, D, indeces)
    
    Inputs:
        * X (array): embedding matrix
        * D (int): dimension of the latent space used for the kernel trick
        * indeces (array): set of indeces for which the normalization constant needs to be computed
        
    Output:
        * Z (array): softmax normalization constant for the set of indeces passed as input
    '''

    n, dim = X.shape

    # create the random projection
    W = np.random.normal(0, 1, (D, dim))
    M = np.zeros((n, 2*D))

    J = X@W.T
    norms = np.sqrt(X**2@np.ones(dim))
    Λ = diags(np.exp(norms**2/2))

    # obtain the high dimensional vectors
    M[:,:D] = Λ@np.sin(J)
    M[:,D:] = Λ@np.cos(J)
    M *= np.sqrt(1/D)

    Z = M[indeces]@np.sum(M, axis = 0)

    return Z


def Normal(μ, σ2, t):
    '''Useful function to get the Normal distribution'''
    
    return 1/np.sqrt(2*np.pi*σ2)*np.exp(-(t - μ)**2/(2*σ2))
