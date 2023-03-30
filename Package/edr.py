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


def CreateEmbedding(Pv, dim = 128, f = None, p0 = None, n_epochs = 20, n_prod = 1, sum_partials = False, k = 1, η = .85, verbose = True, cov_type = 'full'):
    '''
    This function creates a distributed representation of a probability distribution as presented in (Dall'Amico, Belliardo: Efficient distributed representations beyond negative sampling)

    Use: X = CreateEmbedding(Pv)

    Inputs:
        * Pv (list sparse array): the P matrix is provided by the product of all the elements appearing in Pv.from right to left. If only one element is given (Pv = [P]) and `n_prod > 1`, the total probability matrix is given by P^n_prod. If `sum_partials = True` the total matrix `P = (Pv[0] + Pv[1]@Pv[0] + Pv[2]@Pv[1]@Pv[0] + ...)/len(Pv) `. If the resulting matrix P is not a probability matrix (hence it its rows do not sum up to 1), a warning is raised.

    Optional inputs:
        * f (function): this vector specifies the Euclidean norm of each embedding vector. If it is set to `None` (default), then it is considered to be the all ones vector of size n.
        * dim (int): dimension of the embedding. By default set to 128
        * p0 (array): array of size n that specifies the "null model" probability
        * n_epochs (int): number of iterations in the optimization process. By default set to 20
        * n_prod (int): refer to the description of `Pv` for the use of this parameter. Note that if `len(Pv) > 1`, `n_prod` must be set equal to 1 (default value).
        * sum_partials (bool): refer to the description of `Pv` for the use of this parameter. The default value is `False`
        * k (int): order of the GMM approximation. By default set to 8
        * η (float): largest adimissible learning rate. By default set to 0.7.
        * verbose (bool): determines whether the algorithm produces some output for the updates. By default set to True
        * cov_type (string): if 'diag' (default) it computes a diagonal covariance matrix. If 'full' it computes the full covariance matrix. Otherwise it raise a warning.
        
    Output:
        * X (array): solution to the optimization problem for the input weights

    '''

    ### make initial checks and raise warning if needed ###
    F, n, p0 = _RunChecks(Pv, n_prod, cov_type, sum_partials, f, p0)

    ################## run the algorithm ##################


    # initialize the labels for k = 1
    ℓ = np.zeros(n)

    # run the optimization
    if verbose:
        print('Running the optimization for k = 1')

    X = _Optimize(Pv, F, ℓ, p0, n_prod, sum_partials, dim, n_epochs, η, verbose, cov_type)

    # re-run the optimization for k > 1
    if k > 1:
        if verbose:
            print("\n\nComputing the clusters...")

        ℓ = _Clustering(X, k)


        if verbose:
            print("Running the optimization for k = " + str(k))

        X = _Optimize(Pv, F, ℓ, p0, n_prod, sum_partials, dim, n_epochs, η, verbose, cov_type)

    return X


def computeZest(X, indeces, k = 5, return_params = False, cov_type = 'full'):
    '''This function is our implementation of Algorithm 1 and allows to efficiently estimate a set of Z_i values

    Use: Z_vec = computeZest(X, indeces)

    Inputs:
        * X (array): input embedding matrix
        * indeces (array): indices for which Z_i should be computed

    Optional inputs:
        * k (int): order of the mixture model. By default set to 20
        * return_params (bool): if True it returns the values of μ_a, Ω_a, π_a for all a. By default set to False
        * cov_type (string): if 'diag' (default) it returns a diagonal covariance matrix. If 'full' it computes the full covariance matrix. Otherwise a Deprecation Warning is raised

    Output:
        * Z_vec (array): array containing the Z_i values corresponding to the indeces
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
    
    if cov_type == 'diag':
        σ2 = np.stack([np.var((X[ℓ == a])**2, axis = 0) for a in range(k)])
        Zv = np.exp(X[indeces]@μ.T + 0.5*X[indeces]**2@σ2.T)@π
        Ω = [diags(σ2[a]) for a in range(k)]

    elif cov_type == 'full':
        Ω = [np.cov(X[ℓ == a].T) for a in range(k)]
        Zv = np.exp(np.array([X[indeces]@μ[a] + 0.5*(X[indeces] * X[indeces]@Ω[a])@np.ones(dim) for a in range(k)]).T)@π

    else:
        raise DeprecationWarning('Invalid cov_type')


    if return_params:
        return Zv, μ, Ω, π

    else:
        return Zv


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



def _Optimize(Pv, F, ℓ, p0, n_prod, sum_partials, dim, n_epochs, η, verbose, cov_type):
    '''
    This function runs the optimization part of Algorithm 2

    Use: X = _Optimize(Pv, F, ℓ, p0, n_prod, sum_partials, dim, n_epochs, η, verbose, cov_type)

    Inputs:
        * Pv (list sparse array): the P matrix is provided by the product of all the elements appearing in Pv. Note the use of the `n_prod` and `sum_partials` parameters.
        * F (sparse diagonal matrix): the diagonal elements of this matrix are equal to the norms of the embedding vectors.
        * ℓ (array): label assignment
        * p0 (array): array of size n that specifies the "null model" probability
        * n_prod (int): if P can be written as the power or sum of powers of a single matrix, `n_prod` is the largest of these powers.
        * sum_partials (bool): if True, P is written as the sum of powers
        * dim (int): dimension of the embedding.
        * n_epochs (int): number of iterations in the optimization process.
        * η (float): largest learning rate.
        * verbose (bool): determines whether the algorithm produces some output for the updates.
        * cov_type (string): if 'diag' (default) it returns a diagonal covariance matrix. If 'full' it computes the full covariance matrix. Otherwise a Deprecation Warning is raised
        
    Output:
        * X (array): solution to the optimization problem. Its size is n x dim

    '''

    # sample size
    n = Pv[0].shape[0]

    # initialize the weights
    X = np.random.uniform(-1,1, (n, dim))

    # normalize the embedding vectors
    X = F@normalize(X, norm = 'l2', axis = 1)
    f = F.diagonal()
    
    for epoch in range(n_epochs):

        # print update
        if verbose:
            print("[%-25s] %d%%" % ('='*(int((epoch+1)  /n_epochs*25)-1) + '>', (epoch+1)/(n_epochs)*100), end = '\r')


        # compute the gradient
        GRAD = _computeGrad(Pv, X, ℓ, p0, n_prod, sum_partials, cov_type)

        # use the largest possible learing rate
        a = (X * GRAD).sum(-1)
        proj = f**2/(f**2+a)
        if np.sum(proj > 0) > 0:
            proj = proj[proj > 0]
            ηc = np.min([np.min(proj), 1])*η
        else:
            ηc = η

        # update the weights
        X = (1 - ηc)*X - ηc*GRAD
        
        # normalize the embedding vectors
        X = F@normalize(X, norm = 'l2', axis = 1)

    return X


def _computeGrad(Pv, X, ℓ, p0, n_prod, sum_partials, cov_type):
    '''
    This function computes the gradient of the loss function

    Use: GRAD = _computeGrad(Pv, X, ℓ, p0, n_prod, sum_partials, cov_type)

    Inputs:
        * Pv (list sparse array): the P matrix is provided by the product of all the elements appearing in Pv. Note the use of the `walk_length` parameter in the case in which P is the sum of powers of a given matrix.
        * X (array): input weights with respect to which the gradient is computed
        * ℓ (array): label assignment to create the mixture of Gaussians
        * p0 (array): array of size n that specifies the "null model" probability
        * n_prod (int): if P can be written as the power or sum of powers of a single matrix, `n_prod` is the largest of these powers.
        * sum_partials (bool): if True, P is written as the sum of powers
        * cov_type (string): if 'diag' (default) it returns a diagonal covariance matrix. If 'full' it computes the full covariance matrix. Otherwise a Deprecation Warning is raised

    Output:
        * GRAD (array): gradient of the cost function computed in X

    '''

    n, dim = np.shape(X)
    k = len(np.unique(ℓ))

    # compute the parameters
    π = np.array([np.sum(ℓ == a)/n for a in range(k)])
    μ = np.stack([np.mean(X[ℓ == a], axis = 0) for a in range(k)])

    # Z part of the gradient
    if cov_type == 'diag':
        σ2 = np.stack([np.var((X[ℓ == a])**2, axis = 0) for a in range(k)])
        Z = np.exp(X@μ.T + 0.5*X**2@σ2.T)@np.diag(π)
        ZN = diags(1/np.sum(Z, axis = 1))
        Zgrad = ZN.dot(Z.dot(μ) + Z.dot(σ2)*X)

    else:
        Ω = [np.cov((X[ℓ == a]).T) for a in range(k)]
        Z = np.exp(np.array([X@μ[a] + 0.5*(X * X@Ω[a])@np.ones(dim) for a in range(k)]).T)@np.diag(π)
        Zgrad = diags(1/(Z@np.ones(k)))@(Z@μ + np.sum([diags(Z[:,a])@X@Ω[a] for a in range(k)], axis = 0))


    # "energy" part of the gradient
    if sum_partials:
        U, Ut = _computeUsum(Pv, X, n_prod)

    else:
        U, Ut = _computeUprod(Pv, X, n_prod)

    P0 = np.reshape(p0, (n,1))
    u = np.reshape(np.ones(n), (n,1))
    E = u.T@P0

    return -(U + Ut) + Zgrad + (u@(P0.T@X) + P0@(u.T@X))/E


def _computeUsum(Pv, X, n_prod):
    '''
    This function computes the "energetic" contribution of the gradient in the case in which P is written as a sum of products of matrices

    Use: U, Ut = _computeUprod(Pv, X, n_prod)

    Inputs:
        * Pv (list sparse array): The matrix P is given by a sum of the powers of the only elements contained in Pv.
        * X (array): weights with respect to which the gradient is computed
        * n_prod (int): if P can be written as the power or sum of powers of a single matrix, `n_prod` is the largest of these powers.

    Output:
        * U (array): first contribution to the gradient
        * Ut (array): second contribution to the gradient (coming from the transpose)

    '''

    if n_prod > 1:
        Pv = [Pv[0] for i in range(n_prod)]

    # "standard" contribution U
    U = [Pv[0]@X]

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


def _computeUprod(Pv, X, n_prod):
    '''
    This function computes the "energetic" contribution of the gradient in the case in which P is written as a product of matrices

    Use: U, Ut = _computeUprod(Pv, X, n_prod)

    Inputs:
        * Pv (list sparse array): The matrix P is given by a sum of the powers of the only elements contained in Pv.
        * X (array): weights with respect to which the gradient is computed
        * n_prod (int): if P can be written as the power or sum of powers of a single matrix, `n_prod` is the largest of these powers.


    Output:
        * U (array): first contribution to the gradient
        * Ut (array): second contribution to the gradient (coming from the transpose)

    '''

    if n_prod > 1:
        Pv = [Pv[0] for i in range(n_prod)]

    # "standard" contribution U
    U = Pv[0]@X

    for i in range(1, len(Pv)):
        U = Pv[i]@U


    # transpose contribution Ut
    Pv = Pv[::-1]
    Ut = X.T@Pv[0]

    for i in range(1, len(Pv)):
        Ut = Ut@Pv[i]

    return U, Ut.T


def _RunChecks(Pv, n_prod, cov_type, sum_partials, f, p0):
    '''Makes some initial checks on the input variables'''

    # check the consistent use of the n_prod parameter
    if len(Pv) > 1 and n_prod > 1:
        raise DeprecationWarning("Both the value of n_prod and the length of Pv are greater than one")

    n, _ = np.shape(Pv[0])

    # check normalization
    if n_prod > 1:
        P = Pv[0]
        if np.max(np.abs(P@np.ones(n) - np.ones(n))) > 1e-6:
            warnings.warn('The obtained matrix P is not a probability matrix.')
    else:
        v = [np.ones(n)]
        for P in Pv:
            v.append(P@v[-1])

        if sum_partials:
            v = np.sum(v[1:], axis = 0)/len(Pv)
        else:
            v = v[-1]
        if np.max(v - np.ones(n)) > 1e-6:
            warnings.warn('The obtained matrix P is not a probability matrix.')

    # check cov_type
    if not np.isin(cov_type, ['full', 'diag']):
        raise DeprecationWarning('Invalid cov_type')

    # create the diagonal matrix F for vector normalization
    try:
        len(f)
    except:
        f = np.ones(n)

    if len(f) != n:
        raise DeprecationWarning("The provided array f has inconsistent shape with respect to P")
    else:
        F = diags(f/np.mean(f))

    # get and check the p0 vector
    try:
        len(p0)
    except:
        p0 = np.ones(n)

    if len(p0) != n:
        raise DeprecationWarning("The provided array f has inconsistent shape with respect to P")
    

    return F, n, p0
