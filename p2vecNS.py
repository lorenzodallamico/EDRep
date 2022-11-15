import sys
sys.path += ['/home/lorenzo/Scrivania/My_projects/DeepWalk/P2Vec/Package/']  

from scipy.sparse import csr_matrix

from p2vec import *

import warnings
warnings.filterwarnings("ignore")

def CreateEmbeddingNS(Pv, Λ = None, dim = 128, n_epochs = 25, walk_length = 1, negative = 5, η0 = 1., ηfin = 1e-06, 
               γ = 1.0, symmetric = True, verbose = True, scheduler_type = 'linear', shift_and_norm = True):
    '''
    This function implements P2Vec with negative sampling
    
    Use: p2vNS = ComputeEmbeddingNS(Pv)
    
    Inputs:
        * Pv (list sparse array): the P matrix is provided by the product of all the elements appearing in Pv. Note the use of the `walk_length` parameter in the case in which P is the sum of powers of a given matrix.
        
    Optional inputs:
        * Λ (sparse diagonal matrix): weight given to each element in the cost function. If it is set to `None` (default), then it is considered to be the identity matrix
        * dim (int): dimension of the embedding. By default set to 128
        * n_epochs (int): number of iterations in the optimization process. By default set to 20
        * walk_length (int): if P can be written as the sum of powers of a single matrix, `walk_lenght` is the largest of these powers. By default set to 1.
        * negative (int): number of negative samples. By default set to 5
        * η0 (float): initial learning rate. By default set to 1.
        * ηfin (float): final learning rate. By default set to 10**(-6).
        * γ (float): In the case in which P is the sum of powers of matrix M, `γ` is a weight between 0 and 1 multipling M. By default set to 1.
        * symmetric (bool): determines whether to use the symmmetric or the asymmetric version of the algorithm. By default set to True
        * verbose (bool): determines whether the algorithm produces some output for the updates. By default set to True
        * scheduler_type (string): sets the type of the update for the learning parameters. The available options are:
            + 'constant': in this case the learning rate is simply set equal to η0
            + 'linear': in this case the difference between two successive updates is a constant. This is the default value
            + 'exponential': in this case the ratio between two successive updates is a constant.
        * shift_and_norm (bool): is set to True (default) it centers and normalizes the embedding vectors     
    Output:
        * p2v.Φ (array): solution to the optimization problem for the input weights
        * p2v.Ψ (array): solution to the optimization problem for the output weights if the non symmetric version of the algorithm is used
        * p2v.exec_time (float): returns the total execution time
    '''
    
    
    t0 = time.time()

    # check the size of the matrices and raise an error if they do not all have the same dimension and if they are not square
    nv = np.concatenate(np.array([P.shape for P in Pv]))

    if not (nv == nv[0]).all():
        raise DeprecationWarning("The provided sequence of P matrices has invalid shapes")


    # check that the walk_length parameter is properly used
    if walk_length > 1 and len(Pv) > 1:
        raise DeprecationWarning("Both the length of Pv and walk_lenght are greater than one.")


    # -------------------------------------------------------------------------------------

    n = nv[0]
    if Λ == None:
        Λ = diags(np.ones(n))
    else:
        Λ = Λ/np.mean(Λ.diagonal())
        if Λ.shape[0] != n:
            raise DeprecationWarning("The provided matrix Λ has inconsistent shape with respect to P")
            
            
    # --------------------------------------------------------------------------------------
    
    # build the matrix P
    
    # sum of powers
    if walk_length > 1:
        p = Pv[0]
        P  = [p]
        for t in range(1, walk_length):
            P.append(γ*p@P[-1])
        
        P = Λ@np.sum(P, axis = 0)
        P = P/np.mean(P@np.ones(n))

    # matrix product    
    else:
        P = Pv[0]
        for t in range(1, walk_length):
            P = P@Pv[t]
        P = Λ@P
        P = P/np.mean(P@np.ones(n))


    # ----------------------------------------------------------
     
    # run the optimization
    if verbose:
        print('Running the optimization')
    
    if symmetric:
        Φ = _OptimizeSymNS(P, Λ, dim, negative, n_epochs, η0, ηfin, scheduler_type, verbose, shift_and_norm)
        Ψ = 'Not available'
        
    else:
        Φ, Ψ = _OptimizeASymNS(P, Λ, dim, negative, n_epochs, η0, ηfin, scheduler_type, verbose, shift_and_norm)
        
        
    return ReturnValue(Φ, Ψ, 'Not available', 'Not available', 'Not available', 'Not available', 'Not available', time.time() - t0)    


def _OptimizeSymNS(P, Λ, dim, negative, n_epochs, η0, ηfin, scheduler_type, verbose, center_and_norm):
    '''
    This function runs the optimization for the symmetric version of the algorithm using negative sampling
    
    Use: Φ = _OptimizeSymNS(P, Λ, dim, negative, n_epochs, η0, ηfin, scheduler_type, verbose)
    
    Inputs:
        * P (sparse array): this is the matrix P appearing in the cost function
        * Λ (sparse diagonal matrix): weight given to each element in the cost function. If it is set to `None` (default), then it is considered to be the identity matrix
        * dim (int): dimension of the embedding
        * negative (int): number of negative samples
        * n_epochs (int): number of iterations in the optimization process.
        * η0 (float): initial learning rate.
        * ηfin (float): final learning rate.
        * scheduler_type (string): the type of desired update. Refer to the function `_UpdateLearningRate` for further details
        * verbose (bool): determines whether the algorithm produces some output for the updates
        * center_and_norm (bool): if set to True it shifts the mean of the embedding vectors to zero and then sets their norm to one
    Output:
        * Φ (array): solution to the optimization problem. Its size is n x dim
        
    '''

    # sample size
    n = P.shape[0]

    # frequency to choose the negative samples      
    freq = Λ.diagonal()**(3/4)
    freq = freq/np.sum(freq)
    
    # initialize the weights
    Φ = np.random.uniform(-1,1, (n, dim))

    # normalize the weights
    Φ = normalize(Φ, norm = 'l2', axis = 1)

    # generate the probability matrix of the negative samples
    EL = np.array([np.concatenate([[i for j in range(int(P[i].count_nonzero()*negative))] for i in range(n)]),
    np.concatenate([np.random.choice(np.arange(n), size = int(P[i].count_nonzero()*negative),
                    p = freq) for i in range(n)])]).T
    
    N = csr_matrix((np.ones(len(EL[:,0])), (EL[:,0], EL[:,1])), shape = (n,n))
    nn_1 = diags((N@np.ones(n))**(-1))
    N = Λ.dot(nn_1.dot(N))

    # symmetrize the two probability matrices
    P = P + P.T
    N = N + N.T
        
    for epoch in range(n_epochs):
        
        # update the learning rate
        η = UpdateLearningRate(η0, ηfin, epoch, n_epochs, scheduler_type)
        
        # compute the gradient
        GRAD = _computeGradSymNS(P, N, Φ)
             
        # update the weights
        Φ = Φ - η*GRAD

        # print update
        if verbose:
            print("[%-25s] %d%%, η = %f" % ('='*(int((epoch+1)/n_epochs*25)-1) + '>', (epoch+1)/(n_epochs)*100, η), end = '\r')

    
    # shift the mean to zero and normalize the embedding vectors
    if center_and_norm:
        Φ = (Φ.T - np.reshape(np.mean(Φ, axis = 0), (dim, 1))).T        
        Φ = normalize(Φ, norm = 'l2', axis = 1)

    return Φ


def _computeGradSymNS(P, N, Φ):
    '''
    This function computes the gradient for the symmetric version of the algorithm with the negative sampling
    
    Use: GRAD = _computeGradSymNS(P, N, Φ)
    
    Inputs:
        * P (sparse array): the symmetrized P matrix appearing in the cost function
        * N (sparse array): the symmetrized N matrix corresponding to the negative samples
        * Φ (array): input weights with respect to which the gradient is computed
        
    Output:
        * GRAD (array): gradient of the cost function computed in Φ
   
    '''
    
    n, dim = np.shape(Φ)
    
    # compute the matrix of the sigmoid
    sminus = np.sum(Φ[P.nonzero()[0]]*Φ[P.nonzero()[1]], axis = 1)
    σminus = csr_matrix((_sigmoid(-sminus)*np.array(P[P.nonzero()])[0], P.nonzero()), shape = (n,n))
    splus = np.sum(Φ[N.nonzero()[0]]*Φ[N.nonzero()[1]], axis = 1)
    σplus = csr_matrix((_sigmoid(splus)*np.array(N[N.nonzero()])[0], N.nonzero()), shape = (n,n))

    GRAD = (σminus - σplus).dot(Φ)
    
    return -GRAD



def _OptimizeASymNS(P, Λ, dim, negative, n_epochs, η0, ηfin, scheduler_type, verbose, center_and_norm):
    '''
    This function runs the optimization for the non-symmetric version of the algorithm
    
    Use: Φ, Ψ = _OptimizeASymNS(P, Λ, dim, negative, n_epochs, η0, ηfin, scheduler_type, verbose, center_and_norm)
    
    Inputs:
        * P (sparse array): this is the matrix P appearing in the cost function
        * Λ (sparse diagonal matrix): weight given to each element in the cost function. If it is set to `None` (default), then it is considered to be the identity matrix
        * dim (int): dimension of the embedding
        * negative (int): number of negative samples
        * n_epochs (int): number of iterations in the optimization process.
        * η0 (float): initial learning rate.
        * ηfin (float): final learning rate.
        * scheduler_type (string): the type of desired update. Refer to the function `_UpdateLearningRate` for further details
        * verbose (bool): determines whether the algorithm produces some output for the updates
        * center_and_norm (bool): if set to True it shifts the mean of the embedding vectors to zero and then sets their norm to one
    Output:
        * Φ (array): solution to the optimization problem for the input weights. Its size is n x dim
        * Ψ (array): solution to the optimization problem for the output weights. Its size is n x dim
        
    '''
    
    n = P.shape[0]

    # frequency to choose the negative samples      
    freq = Λ.diagonal()**(3/4)
    freq = freq/np.sum(freq)
    

    # initialize the weights
    Φ = np.random.uniform(-1,1, (n, dim))
    Ψ = np.random.uniform(-1,1, (n, dim))

    # normalize the weights
    Φ = normalize(Φ, norm = 'l2', axis = 1)
    Ψ = normalize(Ψ, norm = 'l2', axis = 1)

    # generate the probability matrix of the negative samples
    EL = np.array([np.concatenate([[i for j in range(int(P[i].count_nonzero()*negative))] for i in range(n)]),
    np.concatenate([np.random.choice(np.arange(n), size = int(P[i].count_nonzero()*negative),
                    p = freq) for i in range(n)])]).T
    
    N = csr_matrix((np.ones(len(EL[:,0])), (EL[:,0], EL[:,1])), shape = (n,n))
    nn_1 = diags((N@np.ones(n))**(-1))
    N = Λ.dot(nn_1.dot(N))

    
    for epoch in range(n_epochs):
            
        # set the learning rate
        η = UpdateLearningRate(η0, ηfin, epoch, n_epochs, scheduler_type)
        
        # compute the gradient
        GRADΦ, GRADΨ = _computeGradASymNS(P, N, Φ, Ψ)

        # print update
        if verbose:
            print("[%-25s] %d%%, η = %f" % ('='*(int((epoch+1)/n_epochs*25)-1) + '>', (epoch+1)/(n_epochs)*100, η), end = '\r')
        
        # update the weights
        Φ = Φ - η*GRADΦ
        Ψ = Ψ - η*GRADΨ

    # shift the mean to zero and normalize the embedding vectors
    if center_and_norm:
    
        Φ = (Φ.T - np.reshape(np.mean(Φ, axis = 0), (dim, 1))).T        
        Ψ = (Ψ.T - np.reshape(np.mean(Ψ, axis = 0), (dim, 1))).T
        
        Φ = normalize(Φ, norm = 'l2', axis = 1)
        Ψ = normalize(Ψ, norm = 'l2', axis = 1)
        
    return Φ, Ψ


def _computeGradASymNS(P, N, Φ, Ψ):
    '''
    This function computes the gradient for the symmetric version of the algorithm
    
    Use: GRAD = _computeGradASymNS(P, N, Φ, Ψ)
    
    Inputs:
        * P (sparse array): the symmetrized P matrix appearing in the cost function
        * N (sparse array): the symmetrized N matrix related to the negative samples
        * Φ (array): input weights with respect to which the gradient is computed
        * Ψ (array): output weights with respect to which the gradient is computed
        
    Output:
        * GRADΦ (array): gradient of the cost function with respect to the input weights
        * GRADΨ (array): gradient of the cost function with respect to the output weights
   
    '''

    n, dim = np.shape(Φ)
    
    # compute the matrix of the sigmoid
    sminus = np.sum(Φ[P.nonzero()[0]]*Ψ[P.nonzero()[1]], axis = 1)
    σminus = csr_matrix((_sigmoid(-sminus)*np.array(P[P.nonzero()])[0], P.nonzero()), shape = (n,n))
    splus = np.sum(Φ[N.nonzero()[0]]*Ψ[N.nonzero()[1]], axis = 1)
    σplus = csr_matrix((_sigmoid(splus)*np.array(N[N.nonzero()])[0], N.nonzero()), shape = (n,n))

    GRADΦ = (σminus - σplus)@Ψ
    GRADΨ = (Φ.T@(σminus - σplus)).T
    
    return -GRADΦ, -GRADΨ


def _sigmoid(x):
    return 1/(1+np.exp(-x))