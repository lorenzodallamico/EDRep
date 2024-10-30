# Copyright Lorenzo Dall'Amico and Enrico Maria Belliardo. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import warnings
warnings.simplefilter('always', UserWarning)
import numpy as np
import faiss
from scipy.sparse import diags
from sklearn.preprocessing import normalize
from copy import copy

###########################################
####### Main optimization function ########
###########################################


class EDRep_class:
    def __init__(self, X, Y, ℓ):
        self.X = X
        self.Y = Y
        self.ℓ = ℓ


def CreateEmbedding(Pv, dim = 128, p0 = None, n_epochs = 30, sum_partials = False, k = 1, η = .8, verbose = True, sym = True):
    '''
    This function implemnts the EDRep algorithm

    Use: EDRep = CreateEmbedding(Pv)

    Inputs:
        * Pv (list sparse array): the P matrix is provided by the product of all the elements appearing in Pv.from right to left. If `sum_partials = True` the total matrix `P = (Pv[0] + Pv[1]@Pv[0] + Pv[2]@Pv[1]@Pv[0] + ...)/len(Pv) `. If the resulting matrix P is not a probability matrix (hence it its rows do not sum up to 1), a warning is raised.

    Optional inputs:
        * dim (int): dimension of the embedding. By default set to 128
        * p0 (array): array of size n that specifies the "null model" probability
        * n_epochs (int): number of iterations in the optimization process. By default set to 30
        * sum_partials (bool): refer to the description of `Pv` for the use of this parameter. The default value is `False`
        * k (int): order of the GMM approximation. By default set to 1
        * η (float): largest adimissible learning rate. By default set to 0.8.
        * verbose (bool): determines whether the algorithm produces some output for the updates. By default set to True
        * sym (bool): if True (default) generates a single embedding, while is False it generates two
        
    Output:
        The function returns a class with the following elements:
            * EDRep.X (array): solution to the optimization problem for the input weights
            * EDRep.Y (array): solution to the optimization problem for the input weights
            * EDRep.ℓ (array): label vector

    '''

    ### make initial checks and raise warning if needed ###
    n, m, p0 = _RunChecks(Pv, sum_partials, p0, sym)

    ################## run the algorithm ##################


    # initialize the labels for k = 1
    ℓ = np.zeros(m)

    # run the optimization
    if verbose:
        print('Running the optimization for k = 1')

    X, Y = _Optimize(Pv, ℓ, p0, sum_partials, dim, n_epochs, η, verbose, sym)

    # re-run the optimization for k > 1
    if k > 1:
        if verbose:
            print("\n\nComputing the clusters...")

        ℓ = _Clustering(Y, k)


        if verbose:
            print("Running the optimization for k = " + str(k))

        X, Y = _Optimize(Pv, ℓ, p0, sum_partials, dim, n_epochs, η, verbose, sym)

    return EDRep_class(X, Y, ℓ)


class Zest_class:
    def __init__(self, Zest, ℓ, μ, Ω, π):
        self.Zest = Zest
        self.ℓ = ℓ
        self.μ = μ
        self.Ω = Ω
        self.π = π

def computeZest(X, indeces, k = 5):
    '''This function provides the k order approximation of the Z_i for a set of indes.

    Use: Zest = computeZest(X, indeces, k = 5)

    Inputs:
        * X (array): input embedding matrix
        * indeces (array): indices for which Z_i should be computed

    Optional inputs:
        * k (int): order of the mixture model. By default set to 5
       
    Output:
        The function returns a class with the following elements:
            * Zest (array): array containing the Z_i values corresponding to the indeces
            * ℓ (array): label vector
            * μ (array of size (d x k)): it contains the mean embedding vectors for each class
            * Ω (list of k arrays of size (d x d)): it contains the covariance embedding matrix for each class
            * π (array of size k): size of each class
    '''

    n, dim = np.shape(X)

    if k > 1:
        # estimate the mixture parameters using kmeans
        kmeans = faiss.Kmeans(dim, k, verbose = False, spherical = True)
        kmeans.train(np.ascontiguousarray(X).astype('float32'))
        _, ℓ = kmeans.assign(np.ascontiguousarray(X).astype('float32'))

    else:
        ℓ = np.zeros(n)


    # compute the parameters for each class
    μ = np.array([np.mean(X[ℓ == a], axis = 0) for a in range(k)])
    π = np.array([np.sum(ℓ == a) for a in range(k)])

    # if there is a class with a single element, rerun the algorithm for a k-1
    if np.min(π) == 1:
        return computeZest(X, indeces, k = k-1)
    
    else:
        Ω = [np.cov(X[ℓ == a].T) for a in range(k)]
        Zest = np.exp(np.array([X[indeces]@μ[a] + 0.5*(X[indeces] * X[indeces]@Ω[a])@np.ones(dim) for a in range(k)]).T)@π

        return Zest_class(Zest, ℓ, μ, Ω, π)
    

def computeZ(X, indeces):
    '''This function computes the exact value of Z_i for a set of indeces i

    Use: Z_vec = computeZ(X, indeces)

    Inputs:
        * X (array): input embedding matrix
        * indeces (array): indices for which Z_i should be computed

    Output:
        * Z_vec (array): array containing the Z_i values corresponding to the indeces
    '''

    Z_vec = np.sum(np.exp(X[indeces]@X.T), axis = 1)

    return Z_vec

##########################################
########## Ancillary functions ###########
##########################################

def _Clustering(X, k):
    '''This function generates the label assignment for the Gaussian approximation given the embedding

    Use: ℓ = _Clustering(X, k)

    Inputs:
        * X (array): embedding matrix of size (n x dim)
        * k (int): number of clusters to look for

    Output:
        * ℓ (array): entry-wise label assignment into one of the k clasess.

    '''

    n, dim = np.shape(X)

    kmeans = faiss.Kmeans(dim, k, verbose = False, spherical = True)
    kmeans.train(np.ascontiguousarray(X).astype('float32'))
    _, ℓ = kmeans.assign(np.ascontiguousarray(X).astype('float32'))

    return ℓ



def _Optimize(Pv, ℓ, p0, sum_partials, dim, n_epochs, η, verbose, sym):
    '''
    This function runs the optimization part of the EDRep algorithm

    Use: X, Y = _Optimize(Pv, ℓ, p0, sum_partials, dim, n_epochs, η, verbose, sym)

    Inputs:
        * Pv (list sparse array): the P matrix is provided by the product of all the elements appearing in Pv. Note the use of the `n_prod` and `sum_partials` parameters.
        * ℓ (array): label assignment
        * p0 (array): array of size n that specifies the "null model" probability
        * sum_partials (bool): if True, P is written as the sum of powers
        * dim (int): dimension of the embedding.
        * n_epochs (int): number of iterations in the optimization process.
        * η (float): largest learning rate.
        * verbose (bool): determines whether the algorithm produces some output for the updates.
        * sym (bool): determines whether two calculate one or two embedding matrices
        
    Output:
        * X, Y (array): embedding matrices

    '''

    # sample size
    n = Pv[0].shape[0]
    m = Pv[-1].shape[1]

    eps = η/n_epochs
    
    # initialize the weights
    X = np.random.uniform(-1,1, (n, dim))
    X = normalize(X, norm = 'l2', axis = 1)
            
    if sym:
        Y = copy(X)
    else: 
        Y = np.random.uniform(-1,1, (m, dim))
        Y = normalize(Y, norm = 'l2', axis = 1)
            

    for epoch in range(n_epochs):
        
        # print update
        if verbose:
            print("[%-25s] %d%%" % ('='*(int((epoch+1)  /n_epochs*25)-1) + '>', (epoch+1)/(n_epochs)*100), end = '\r')

        # compute the gradient
        GRADX, GRADY = _computeGrad(Pv, X, Y, ℓ, p0, sum_partials)
        
        # update the weights
        if sym:
            GRAD = GRADX + GRADY
            D = diags((GRAD * X).sum(axis = 1))
            GRAD = GRAD - D@X
            GRAD = normalize(GRAD, norm = 'l2', axis = 1)
            X = np.sqrt(1-η**2)*X - η*GRAD
            Y = copy(X)

        else:
            DX = diags((GRADX * X).sum(axis = 1))
            GRADX = GRADX - DX@X
            GRADX = normalize(GRADX, norm = 'l2', axis = 1)

            DY = diags((GRADY * Y).sum(axis = 1))
            GRADY = GRADY - DY@Y
            GRADY = normalize(GRADY, norm = 'l2', axis = 1)
            
            X = np.sqrt(1-η**2)*X - η*GRADX
            Y = np.sqrt(1-η**2)*Y - η*GRADY

        η = η - eps
            
    return X,Y


def _computeGrad(Pv, X, Y, ℓ, p0, sum_partials):
    '''
    This function computes the gradient of the loss function

    Use: GRADX, GRADY = _computeGrad(Pv, X, Y, ℓ, p0, sum_partials)

    Inputs:
        * Pv (list sparse array): the P matrix is provided by the product of all the elements appearing in Pv. Note the use of the `walk_length` parameter in the case in which P is the sum of powers of a given matrix.
        * X (array): input weights with respect to which the gradient is computed
        * Y (array): output wieghts with respect to which the gradient is computed
        * ℓ (array): label assignment to create the mixture of Gaussians
        * p0 (array): array of size n that specifies the "null model" probability
        * sum_partials (bool): if True, P is written as the sum of powers
        
    Output:
        * GRADX, GRADY (array): gradient of the cost function computed in X and Y

    '''

    n, dim = np.shape(X)
    m, _ = np.shape(Y)
    k = len(np.unique(ℓ))

    # compute the parameters
    π = np.array([np.sum(ℓ == a)/n for a in range(k)])
    μ = np.stack([np.mean(Y[ℓ == a], axis = 0) for a in range(k)])

    # Z part of the gradient
    Ω = [np.cov((Y[ℓ == a]).T) for a in range(k)]
    Z = np.exp(np.array([X@μ[a] + 0.5*(X * X@Ω[a])@np.ones(dim) for a in range(k)]).T)@np.diag(π)
    Zgrad = diags(1/(Z@np.ones(k)))@(Z@μ + np.sum([diags(Z[:,a])@X@Ω[a] for a in range(k)], axis = 0))


    # "energy" part of the gradient
    if sum_partials:
        U, Ut = _computeUsum(Pv, X, Y)

    else:
        U, Ut = _computeUprod(Pv, X, Y)

    P0 = np.reshape(p0, (m,1))
    u = np.reshape(np.ones(n), (n,1))
    E = np.sum(p0)

    return -U + Zgrad + (u@(P0.T@Y))/E, -Ut + P0@(u.T@X)/E



def _computeUsum(Pv, X, Y):
    '''
    This function computes the "energetic" contribution of the gradient in the case in which P is written as a sum of products of matrices

    Use: U, Ut = _computeUsum(Pv, X, Y)

    Inputs:
        * Pv (list sparse array): The matrix P is given by a sum of the powers of the only elements contained in Pv.
        * X (array): input weights with respect to which the gradient is computed
        * Y (array): output weights with respect to which the gradient is computed
       
    Output:
        * U (array): first contribution to the gradient
        * Ut (array): second contribution to the gradient (coming from the transpose)

    '''

    # "standard" contribution U
    U = [Pv[0]@Y]

    for i in range(1, len(Pv)):
        U.append(Pv[i]@U[-1])


    # transpose contribution Ut
    Pv = Pv[::-1]
    Ut = [X.T@Pv[0]]

    for i in range(1, len(Pv)):
        Ut.append(Ut[-1]@Pv[i])

    U = np.sum(U, axis = 0)/len(Pv)
    Ut = np.sum(Ut, axis = 0)/len(Pv)

    return U, Ut.T


def _computeUprod(Pv, X, Y):
    '''
    This function computes the "energetic" contribution of the gradient in the case in which P is written as a product of matrices

    Use: U, Ut = _computeUprod(Pv, X, Y)

    Inputs:
        * Pv (list sparse array): The matrix P is given by a sum of the powers of the only elements contained in Pv.
        * X (array): input weights with respect to which the gradient is computed
        * Y (array): output weights with respect to which the gradient is computed

    Output:
        * U (array): first contribution to the gradient
        * Ut (array): second contribution to the gradient (coming from the transpose)

    '''

    # "standard" contribution U
    U = Pv[0]@Y

    for i in range(1, len(Pv)):
        U = Pv[i]@U


    # transpose contribution Ut
    Pv = Pv[::-1]
    Ut = X.T@Pv[0]

    for i in range(1, len(Pv)):
        Ut = Ut@Pv[i]

    return U, Ut.T


def _RunChecks(Pv, sum_partials, p0, sym):
    '''Makes some initial checks on the input variables'''

    # check the consistency of the input matrices sizes
    for i in range(len(Pv)-1):
        if Pv[i].shape[1] != Pv[i+1].shape[0]:
            raise DeprecationWarning('The input matrices shapes are inconsistent')
        

    # check the consistency with the use of the `sym` parameter
    n = Pv[0].shape[0]
    m = Pv[-1].shape[1]
    if (n != m) and sym:
        raise DeprecationWarning('Invalid sym type: the probability matrix is rectangular and sym = False is required as input')
    
    # check the consistency with the use of `sum_partials` parameter
    if sum_partials:
        for P in Pv:
            a, b = P.shape
            if a != b:
                raise DeprecationWarning('`sum_partials = True` can only be set if all matrices are square')

    # check normalization
    v = [np.ones(m)]
    for P in Pv:
        v.append(P@v[-1])

    if sum_partials:
        v = np.sum(v[1:], axis = 0)/len(Pv)
    else:
        v = v[-1]
    if np.max(v - np.ones(n)) > 1e-6:
        warnings.warn('The obtained matrix P is not a probability matrix.')

    # get and check the p0 vector
    try:
        len(p0)
    except:
        p0 = np.ones(m)

    if len(p0) != m:
        raise DeprecationWarning("The provided array f has inconsistent shape with respect to P")
    

    return n, m, p0
