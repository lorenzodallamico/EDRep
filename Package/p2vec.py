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


import numpy as np
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture
import faiss
from scipy.sparse import diags
import matplotlib.pyplot as plt
import time

# -----------------------------------------------


class ReturnValue:
    def __init__(self, Φ, ℓin, π, μ, σ2, exec_time, loss):
        self.Φ = Φ
        self.ℓin = ℓin
        self.π = π
        self.μ = μ
        self.σ2 = σ2
        self.exec_time = exec_time
        self.loss = loss
        
        
    def computeZest(self, i):
        '''
        This function computes the estimated partition function for the element i
        
        Use: Z = p2v.computeZest(i)
        
        Input: i (int), index of the element for which the partition function should be computed
        
        Output: Z (float), partition function
        '''
        
        Φ = self.Φ
        μ = self.μ
        σ2 = self.σ2
        π = self.π
        
        return np.exp(Φ[i]@μ.T + 0.5*(Φ[i]**2)@σ2.T)@π
    
    
    def computeZ(self, i):
        '''
        This function computes the exact partition function for the element i
        
        Use: Z = p2v.computeZ(i)
        
        Input: i (int), index of the element for which the partition function should be computed
        
        Output: Z (float), partition function
        '''
        
        Φ = self.Φ
        
        return np.sum(np.exp(Φ[i]@Φ.T))
        
        
        
        
    def plotDistr(self, i,  *args, **kwargs):
        '''
        This function plots the histogram of the scalar product entering the partition function and compares it to the theoretical value
        
        Use: p2v.plotDistr(i)
        
        Input:
            * i (int): index of the element the plot is referred to
            
        Optional inputs:
            * bins (int): number of bins in the histogram
            * color (string): color of the histogram
            * edgecolor (string): color of the histogram edges
            * figsize (tuple): specifies the figure size
            * linecolor (string): color of the theoretical line
        '''
        
        bins = kwargs.get('bins', 60)
        color = kwargs.get('color', 'chartreuse')
        edgecolor = kwargs.get('edgecolor', 'white')
        figsize = kwargs.get('figsize', (7,5))
        linecolor = kwargs.get('linecolor', 'k')
        
        Φ = self.Φ
        μ = self.μ
        σ2 = self.σ2
        π = self.π
        n = np.sum(π)
        π = π/n
        k = np.shape(μ)[0]       
        
        plt.figure(figsize = figsize)
        
        plt.hist(Φ[i]@Φ.T, bins = bins, color = color, edgecolor = edgecolor, density = True)
            
            
        m = [Φ[i]@μ[a] for a in range(k)]
        v = [(Φ[i])**2@σ2[a] for a in range(k)]
         
        t = np.linspace(np.min(m) - 4*np.sqrt(np.max(v)), np.max(m) + 4*np.sqrt(np.max(v)),1000)
        y = np.zeros(len(t))
        
        for a in range(k):
            if v[a] > 0:
                y += π[a]*np.exp(-(t-m[a])**2/(2*v[a]))/np.sqrt(2*np.pi*v[a])
            
        plt.plot(t, y, color = linecolor)
        
        plt.show();

        return
    

# CHECK THE MATRIX SIZES
def CreateEmbedding(Pv, Λ = None, dim = 128, n_epochs = 20, n_epochs_before_rescheduling = 1, walk_length = 1, k = 8, η0 = 1., 
                    ηfin = 10**(-6), γ = 1., verbose = True, scheduler_type = 'linear', 
                    recompute_labels = False, est_method = 'KM', optimizer = None, seed = None, normalizeΦ = True, rescaleΦ = None, 
                    clipping = False, compute_loss = False):
    '''
    This is the implementation of P2Vec
    
    Use: p2v = CreateEmbedding(P)
    
    Inputs:
        * Pv (list sparse array): the P matrix is provided by the product of all the elements appearing in Pv. Note the use of the `walk_length` parameter in the case in which P is the sum of powers of a given matrix.
        
    Optional inputs:
        * Λ (sparse diagonal matrix): weight given to each element in the cost function. If it is set to `None` (default), then it is considered to be the identity matrix
        * dim (int): dimension of the embedding. By default set to 128
        * n_epochs (int): number of iterations in the optimization process. By default set to 20
        * n_epochs_before_rescheduling (int): number of epochs before updating the learning rate. By default set to 1.
        * walk_length (int): if P can be written as the sum of powers of a single matrix, `walk_lenght` is the largest of these powers. By default set to 1.
        * k (int): order of the EM approximation. By default set to 8
        * η0 (float): initial learning rate. By default set to 1.
        * ηfin (float): final learning rate. By default set to 10**(-6).
        * γ (float): In the case in which P is the sum of powers of matrix M, `γ` is a weight between 0 and 1 multipling M. By default set to 1.
        * verbose (bool): determines whether the algorithm produces some output for the updates. By default set to True
        * scheduler_type (string): sets the type of the update for the learning parameters. The available options are:
            + 'constant': in this case the learning rate is simply set equal to η0
            + 'linear': in this case the difference between two successive updates is a constant. This is the default value
            + 'exponential': in this case the ratio between two successive updates is a constant.
            + 'mixed': it does n-3 rescheduling steps with a linear decay up to η = 0.01 and then it continues with an exponential decay.
            
        * recompute_labels (bool): if set to True it will recompute the labels obtained with the Gaussian mixture model after convergence, otherwise the output will give the ones computed before the last optimization. By default set to False
        * est_method (string): determines the algorithm to estimate the mixture of Gaussian parameters. The options are
            + 'KM': uses K-Means algorithm (default)
            + 'EM': uses expecation maximization
        * optimizer: select optimizer different from default gradient descent (default = None). The available options are:
            + 'AdamW': perform AdamW optimization with default parameters
        * seed (int): initialize random state for reproducibility. By default is set to None
        * normalizeΦ (bool): normalize the embeddings matrix after each epoch of the optimization process. Default value is True
        * rescaleΦ (array): if normalizeΦ = True, rescale each row of Φ by a value proportional to rescaleΦ. Default is set to None
        * clipping (bool): gradient clipping. Threshold is set as sqrt(dim). Default value is False 
        * compute_loss (bool): output loss function (up to a constant) for each iteration. Default is set to False
        

            
    Output:
        * p2v.Φ (array): solution to the optimization problem for the input weights
        * p2v.Ψ (array): solution to the optimization problem for the output weights if the non symmetric version of the algorithm is used
        * p2v.ℓin (array): label assignment obtained on the weights
        * p2v.π (array): this vector contains the number of elements in each clusters
        * p2v.μ (array): centers of the weights for each cluster
        * p2v.σ2 (array): variances of the weights for each cluster
        * p2v.exec_time (float): returns the total execution time
        * p2v.loss (array): if compute_loss = True, returns the loss function (up to a constant) for each epoch

    The class further outputs three functions
    * p2v.computeZest
    * p2v.computeZ
    * p2v.plotDistr

    To see their documentation call (for instance) `p2v.computeZest?`.
    '''
    
    t0 = time.time()

    # check the size of the matrices and raise an error if they do not all have the same dimension and if they are not square
    nv = np.concatenate(np.array([P.shape for P in Pv]))

    # if not (nv == nv[0]).all():
    #     raise Exception("The provided sequence of P matrices has invalid shapes")


    # check that the walk_length parameter is properly used
    if walk_length > 1 and len(Pv) > 1:
        raise Exception("Both the length of Pv and walk_lenght are greater than one.")
    
    # check rescaleΦ dimension, if used
    if (rescaleΦ is not None) and (rescaleΦ.shape[0] != nv[0]):
        raise Exception("Inconsistent rescaleΦ dimension.")

    # -------------------------------------------------------------------------------------

    n = nv[0]
    if Λ == None:
        Λ = diags(np.ones(n))
    else:
        # if Λ.shape[0] != n:
        #     raise Exception("The provided matrix Λ has inconsistent shape with respect to P")
        # else:
        Λ = Λ.diagonal()
        Λ = diags(Λ/np.mean(Λ)) 

    # initialize the labels for k = 1
    ℓ = np.zeros(n)

    # symmetric version of the algorithm
    if verbose:
        print('Running the optimization for k = 1')
        
    # run the optimization
    Φ, loss = _OptimizeSym(Pv, Λ, ℓ, walk_length, dim, n_epochs, n_epochs_before_rescheduling, η0, ηfin, γ, scheduler_type, verbose, optimizer, seed, normalizeΦ, rescaleΦ, clipping, compute_loss)

    if k > 1:

        if verbose:
            print("\n\nComputing the clusters...")

        # estimate the mixture of Gaussian parameters
        if est_method == 'KM':
            kmeans = faiss.Kmeans(dim, k, verbose = False)
            kmeans.train(np.ascontiguousarray(Φ).astype('float32'))
            _, ℓ = kmeans.assign(np.ascontiguousarray(Φ).astype('float32'))

        elif est_method == 'EM':
            gm = GaussianMixture(n_components = k, covariance_type = 'diag').fit(Φ)
            ℓ = gm.predict(Φ)

        else:
            raise Exception("The selected estimation method is not valid")

        if verbose:
            print("Running the optimization for k = " + str(k))
        
        # re-run the optimization
        Φ, loss = _OptimizeSym(Pv, Λ, ℓ, walk_length, dim, n_epochs, n_epochs_before_rescheduling, η0, ηfin, γ, scheduler_type, verbose, optimizer, seed, normalizeΦ, rescaleΦ, clipping, compute_loss)
        
    if verbose:
        print("\n\nComputing the parameters values...")
        
    μ = np.array([np.mean(Φ[ℓ == a], axis = 0) for a in range(k)])
    σ2 = np.array([np.var(Φ[ℓ == a], axis = 0) for a in range(k)])
    π = np.array([np.sum(ℓ == a) for a in range(k)])
    
    if recompute_labels:
        if est_method == 'KM':
            kmeans = faiss.Kmeans(dim, k, verbose = False)
            kmeans.train(np.ascontiguousarray(Φ).astype('float32'))
            _, ℓin = kmeans.assign(np.ascontiguousarray(Φ).astype('float32'))

        elif est_method == 'EM':
            gm = GaussianMixture(n_components = k, covariance_type = 'diag').fit(Φ)
            ℓin = gm.predict(Φ)
    else:
        ℓin = ℓ 

    return ReturnValue(Φ, ℓin, π, μ, σ2, time.time() - t0, loss)

######################################################################################
######################################################################################
######################################################################################


def UpdateLearningRate(η0, ηfin, epoch, n_epochs, n_epochs_before_rescheduling, scheduler_type):
    '''
    This function is used to update the learning rate at each iteration.
    
    Use: η = UpdateLearningRate(η0, ηfin, epoch, n_epochs, scheduler_type)
    
    Inputs:
        * η0 (float): the learning rate at the first iteration
        * ηfin (float): the learning rate at the last iteration
        * epoch (int): the current epoch
        * n_epochs (int): the total number of epochs
        * n_epochs_before_rescheduling (int): number of epochs after which the learning rate is updated
        * scheduler_type (string): the type of desired update. The available types are:
            + 'constant': in this case the learning rate is simply set equal to η0
            + 'linear': in this case the difference between two successive updates is a constant
            + 'exponential': in this case the ratio between two successive updates is a constant
            + 'mixed': it does n-3 rescheduling steps with a linear decay up to η = 0.01 and then it continues with an exponential decay
            
    Output:
        * η (float): the updated learning rate
    '''
    n_epochs_ = (n_epochs-1) // n_epochs_before_rescheduling
    epoch_ = epoch // n_epochs_before_rescheduling

    if n_epochs_ > 0 :
        if scheduler_type == 'exponential':
            c = (ηfin/η0)**(1/(n_epochs_))
            η = η0*c**epoch_
            
        elif scheduler_type == 'constant':
            η = η0
            
        elif scheduler_type == 'linear':
            c = (ηfin - η0)/(n_epochs_ )
            η = η0 + c*epoch_

        elif scheduler_type == 'mixed':
            if n_epochs_ <= 3:
                raise Exception("Too few iterations for scheduler_type = 'mixed' for the specified n_epochs_before_rescheduling")
            
            elif epoch_ <= n_epochs_ - 3:
                c = (0.01 - η0)/(n_epochs_ - 3)
                η = η0 + c*epoch_

            else:
                c = (ηfin/0.01)**(1/(3-1))
                η = 0.01*c**(epoch_ - n_epochs_ + 3)

        
        else:
            raise Exception('The scheduler_type variable is not valid')
    else:
        η = η0
        
    return η

def _OptimizeSym(Pv, Λ, ℓ, walk_length, dim, n_epochs, n_epochs_before_rescheduling, η0, ηfin, γ, scheduler_type, verbose, optimizer, seed, normalizeΦ, rescaleΦ, clipping, compute_loss):
    '''
    This function runs the optimization for the symmetric version of the algorithm.
    
    Use: Φ = _OptimizeSym(Pv, Λ, ℓ, walk_length, dim, n_epochs, η0, ηfin, γ, scheduler_type, verbose, optimizer)
    
    Inputs:
        * Pv (list sparse array): the P matrix is provided by the product of all the elements appearing in Pv. Note the use of the `walk_length` parameter in the case in which P is the sum of powers of a given matrix.
        * Λ (sparse diagonal matrix): weight given to each element in the cost function. 
        * ℓ (array): label assignment to create the mixture of Gaussians.
        * walk_length (int): if P can be written as the sum of powers of a single matrix, `walk_lenght` is the largest of these powers.
        * dim (int): dimension of the embedding.
        * n_epochs (int): number of iterations in the optimization process.
        * n_epochs_before_rescheduling (int): number of epochs before updating the learning rate. 
        * η0 (float): initial learning rate.
        * ηfin (float): final learning rate.
        * γ (float): In the case in which P is the sum of powers of matrix M, `γ` is a weight between 0 and 1 multipling M.
        * scheduler_type (string): the type of desired update. Refer to the function `UpdateLearningRate` for further details.
        * verbose (bool): determines whether the algorithm produces some output for the updates.
        * optimizer (string): choose between default gradient descent or 'AdamW'. 
        * seed (int): initialize random state for reproducibility.
        * normalizeΦ (bool): normalize the embeddings matrix after each epoch of the optimization process.
        * rescaleΦ (array): if normalizeΦ = True, rescale each row of Φ by a value proportional to rescaleΦ.
        * clipping (bool): gradient clipping. Threshold is set as sqrt(dim).
        * compute_loss (bool): output loss function (up to a constant) for each iteration.
    Output:
        * Φ (array): solution to the optimization problem. Its size is n x dim.
        * loss (array): if compute_loss = True, loss function (up to a constant) for each epoch.
        
    '''
    # loss
    loss = [] if compute_loss else None
    # sample size
    n = Pv[0].shape[0]
    
    # initialize the weights
    if seed:
        np.random.seed(seed)
    Φ = np.random.uniform(-1,1, (n, dim))
    
    # normalize the embedding vectors
    Φ = normalize(Φ, norm = 'l2', axis = 1)

    # run optimization    
    if optimizer == "AdamW":
        Φ, loss = AdamWOptimizerSym(Φ, Pv, Λ, ℓ, walk_length, dim, n_epochs, n_epochs_before_rescheduling, η0, ηfin, γ, scheduler_type, verbose, normalizeΦ, rescaleΦ, clipping, compute_loss)
    
    else:
        for epoch in range(n_epochs):
            
            # update learning rate
            if epoch % n_epochs_before_rescheduling == 0:
                η = UpdateLearningRate(η0, ηfin, epoch, n_epochs, n_epochs_before_rescheduling, scheduler_type)
            
            # print update
            if verbose:
                print("[%-25s] %d%%, η = %f" % ('='*(int((epoch+1)/n_epochs*25)-1) + '>', (epoch+1)/(n_epochs)*100, η), end = '\r')

            # compute loss
            if compute_loss:
                loss.append(_computeLossSym(Pv, Λ, Φ, ℓ, γ, walk_length))
            
            # compute the gradient and update the weights
            GRAD = _computeGradSym(Pv, Λ, Φ, ℓ, γ, walk_length, dim, clipping)
            Φ = Φ - η*GRAD

            # shift the mean to zero
            Φ = (Φ.T - np.reshape(np.mean(Φ, axis = 0), (dim, 1))).T 

            # normalize the embedding vectors
            if normalizeΦ:
                Φ = normalize(Φ, norm = 'l2', axis = 1)
            
                if rescaleΦ is not None:
                    rescaleΦ_ = rescaleΦ/np.mean(rescaleΦ)   
                    Φ = diags(rescaleΦ_)@Φ

    loss = np.array(loss)
    return Φ, loss

def _computeLossSym(Pv, Λ, Φ, ℓ, γ, walk_length):

    n = Pv[0].shape[0]
    k = len(np.unique(ℓ))

    # compute the parameters
    π = np.array([np.sum(ℓ == a)/n for a in range(k)])
    μ = np.stack([np.mean(Φ[ℓ == a], axis = 0) for a in range(k)])
    σ2 = np.stack([np.var((Φ[ℓ == a])**2, axis = 0) for a in range(k)])

    # "energy" part of the gradient
    if walk_length > 1:
        U, _ = _computeUsum(Pv, Λ, Φ, γ, walk_length)

    else:
        U, _ = _computeUprod(Pv, Λ, Φ)


    Z = np.exp(Φ@μ.T + 0.5*Φ**2@σ2.T)@np.diag(π)

    loss = -np.trace(Φ.T@U)  + np.log(np.sum(Z))
    
    return loss

def _computeGradSym(Pv, Λ, Φ, ℓ, γ, walk_length, dim, clipping):
    '''
    This function computes the gradient for the symmetric version of the algorithm
    
    Use: GRAD = _computeGradSym(Pv, Λ, Φ, ℓ, γ, walk_length)
    
    Inputs:
        * Pv (list sparse array): the P matrix is provided by the product of all the elements appearing in Pv. Note the use of the `walk_length` parameter in the case in which P is the sum of powers of a given matrix.
        * Λ (sparse diagonal matrix): weight given to each element in the cost function. 
        * Φ (array): input weights with respect to which the gradient is computed
        * ℓ (array): label assignment to create the mixture of Gaussians
        * γ (float): In the case in which P is the sum of powers of matrix M, `γ` is a weight between 0 and 1 multipling M.
        * walk_length (int): if P can be written as the sum of powers of a single matrix, `walk_lenght` is the largest of these powers.
        * dim (int): dimension of the embedding.
        * clipping (bool): gradient clipping. Threshold is set as sqrt(dim).
        
    Output:
        * GRAD (array): gradient of the cost function computed in Φ
   
    '''

    n = Pv[0].shape[0]
    k = len(np.unique(ℓ))

    # compute the parameters
    π = np.array([np.sum(ℓ == a)/n for a in range(k)])
    μ = np.stack([np.mean(Φ[ℓ == a], axis = 0) for a in range(k)])
    σ2 = np.stack([np.var((Φ[ℓ == a])**2, axis = 0) for a in range(k)])

    # "energy" part of the gradient
    if walk_length > 1:
        U, Ut = _computeUsum(Pv, Λ, Φ, γ, walk_length)

    else:
        U, Ut = _computeUprod(Pv, Λ, Φ)


    Z = np.exp(Φ@μ.T + 0.5*Φ**2@σ2.T)@np.diag(π)
    ZN = diags(Λ.diagonal()/np.sum(Z, axis = 1))

    GRAD = -U - Ut + ZN.dot(Z.dot(μ) + Z.dot(σ2)*Φ)

    if clipping:
        GradNorm = np.sqrt(GRAD**2@np.ones(dim))
        idx = GradNorm > np.sqrt(dim)
        v = np.ones(n)
        v[idx] = (np.sqrt(dim)**-1)*GradNorm[idx]
        V = diags(v**(-1))
        GRAD = V.dot(GRAD)
    
    return GRAD

def _computeUsum(Pv, Λ, Φ, γ, walk_length):
    '''
    This function computes the "energetic" contribution of the gradient in the case in which P is written as a sum of powers.
    
    Use: U, Ut = _computeUsum(Pv, Λ, Φ, γ, walk_length)
    
    Inputs:
        * Pv (list sparse array): The matrix P is given by a sum of the powers of the only elements contained in Pv.
        * Λ (sparse diagonal matrix): weight given to each element in the cost function. 
        * Φ (array): weights with respect to which the gradient is computed.
        * γ (float): a weight between 0 and 1 multipling the only element contained in Pv.
        * walk_length (int): the largest matrix power considered to build P.
        
    Output:
        * U (array): first contribution to the gradient.
        * Ut (array): second contribution to the gradient (coming from the transpose).
        
    '''
    
    P = Pv[0]
    Φt = Φ.T@Λ

    U = [P@Φ]
    Ut = [Φt@P]

    for t in range(1,walk_length):
        U.append(γ*P@U[-1])
        Ut.append(γ*Ut[-1]@P)

    U = Λ@(np.sum(U, axis = 0))
    Ut = np.sum(Ut, axis = 0)

    
    return U, Ut.T


def _computeUprod(Pv, Λ, Φ):
    '''
    This function computes the "energetic" contribution of the gradient in the case in which P is written as a product of matrices.
    
    Use: U, Ut = _computeUprod(Pv, Λ, Φ)
    
    Inputs:
        * Pv (list sparse array): The matrix P is given by a sum of the powers of the only elements contained in Pv.
        * Λ (sparse diagonal matrix): weight given to each element in the cost function. 
        * Φ (array): weights with respect to which the gradient is computed.
        
    Output:
        * U (array): first contribution to the gradient.
        * Ut (array): second contribution to the gradient (coming from the transpose).
        
    '''

    # transpose contribution Ut
    Ut = Φ.T@Λ

    for i in range(len(Pv)):
        Ut = Ut@Pv[i]

    # standard contribution U
    Pv = Pv[::-1]
    U = Pv[0]@Φ
    
    for i in range(1, len(Pv)):
        U = Pv[i]@U
            
    U = Λ@U
        
    return U, Ut.T

def AdamWOptimizerSym(Φ, Pv, Λ, ℓ, walk_length, dim, n_epochs, n_epochs_before_rescheduling, η0, ηfin, γ, scheduler_type, verbose, normalizeΦ, rescaleΦ, clipping, compute_loss):
    '''
    This function performs AdamW optimization for the symmetric version of the algorithm.
    
    Use: Φ = AdamWOptimizer(Φ, Pv, Λ, ℓ, walk_length, dim, n_epochs, n_epochs_before_rescheduling, η0, ηfin, γ, scheduler_type, verbose, normalizeΦ, rescaleΦ, clipping, compute_loss)
    
    Inputs:
        * Φ (array): initial weights to be optimized
        * Pv (list sparse array): The matrix P is given by a sum of the powers of the only elements contained in Pv.
        * Λ (sparse diagonal matrix): weight given to each element in the cost function. 
        * ℓ (array): label assignment to create the mixture of Gaussians
        * walk_length (int): if P can be written as the sum of powers of a single matrix, `walk_lenght` is the largest of these powers.
        * dim (int): dimension of the embedding. 
        * n_epochs (int): number of iterations in the optimization process.
        * n_epochs_before_rescheduling (int):
        * η0 (float): initial learning rate.
        * ηfin (float): final learning rate.
        * γ (float): In the case in which P is the sum of powers of matrix M, `γ` is a weight between 0 and 1 multipling M.
        * scheduler_type (string): the type of desired update. Refer to the function `UpdateLearningRate` for further details.
        * verbose (bool): determines whether the algorithm produces some output for the updates.
        * normalizeΦ (bool): normalize the embeddings matrix after each epoch of the optimization process.
        * rescaleΦ (array): if normalizeΦ = True, rescale each row of Φ by a value proportional to rescaleΦ.
        * clipping (bool): gradient clipping. Threshold is set as sqrt(dim).
        * compute_loss (bool): output loss function (up to a constant) for each iteration.


    Output:
        * Φ (array): solution to the optimization problem for the input weights. Its size is n x dim
        * loss (array): if compute_loss = True, loss function (up to a constant) for each epoch.
        
    '''
    
    #hyperparameters
    alpha = 1.
    beta_1 = 0.9
    beta_2 = 0.999
    eps = 10**-8
    w = 0.1

    #initialize empty loss 
    loss = [] if compute_loss else None

    #initialize empty arrays and time
    m = np.zeros(shape = (n, dim))
    v = np.zeros(shape = (n, dim))
    t = 0
    for epoch in range(n_epochs):
        if epoch % n_epochs_before_rescheduling == 0:
            η = UpdateLearningRate(η0, ηfin, epoch, n_epochs, n_epochs_before_rescheduling, scheduler_type)

        # print update
        if verbose:
            print("[%-25s] %d%%, η = %f" % ('='*(int((epoch+1)/n_epochs*25)-1) + '>', (epoch+1)/(n_epochs)*100, η), end = '\r')

        # compute loss
        if compute_loss:
            loss.append(_computeLossSym(Pv, Λ, Φ, ℓ, γ, walk_length))
        
        # compute the gradient and update the weights
        t +=1 
        GRAD = _computeGradSym(Pv, Λ, Φ, ℓ, γ, walk_length, dim, clipping)   
        m = beta_1*m + (1-beta_1)*GRAD
        v = beta_2*v + (1-beta_2)*GRAD**2
        m_hat = m/(1-beta_1**t)
        v_hat = v/(1-beta_2**t)
        Φ -= η*(alpha*m_hat/(np.sqrt(v_hat) + eps) +w*Φ)

        # shift the mean to zero
        Φ = (Φ.T - np.reshape(np.mean(Φ, axis = 0), (dim, 1))).T 

        # normalize the embedding vectors
        if normalizeΦ:
            Φ = normalize(Φ, norm = 'l2', axis = 1)

            if rescaleΦ is not None:
                rescaleΦ_ = rescaleΦ/np.mean(rescaleΦ)   
                Φ = diags(rescaleΦ_)@Φ
    
    loss = np.array(loss)
    return Φ, loss