import sys
sys.path += ['/home/lorenzo/Scrivania/My_projects/DeepWalk/P2Vec/Package/']  

import pandas as pd
import seaborn as sns

from p2vec import *

import warnings
warnings.filterwarnings("ignore")


class TestReturn:
    def __init__(self, df, time_est, time_exact):
        self.df = df
        self.time_est = time_est
        self.time_exact = time_exact
        
    def PlotTest(self):
        
        sns.violinplot(x = "k", y = "err", data = self.df)
        plt.show();


def TestApprox(Φ, indeces, kv, Ψ = None, verbose = True):
    '''This function tests our approximation of softmax for a set of indeces and values of k
    
    Use: ta = TestApprox(Φ, indeces, kv)
    
    Inputs:
        * Φ (array): embedding vectors
        * indeces (array): indeces for which the partition function will be computed
        * kv (array): list of integer values of k for which the aprroximation will be evaluated
        
    Optional inputs:
        * Ψ (array): output weights used in the non symmetric version of the algorithm. By defaul `None`
        * verbose (bool): is True (default) it will print the progress bar
        
    Output:
        * ta (class)
            + ta.df (pandas dataframe): this dataframe contains two columns: 
                - ϵ: relative error with respect to the exact value of Z 
                - k: value of k corresponding to that estimation
            + ta.time_est (array): time to obtain all the estimates of Z for each value of k
            + ta.time_exact (float): time to compute the exact values of Z
            + ta.PlotTest(): produces violin plots of the error for each value of k
    
    '''
    
    Zest = []
    time_est = []
    
    # estimate Z for different values of k
    for i, k in enumerate(kv):
        t0 = time.time()
        Zest.append(ComputeZest(Φ, indeces , k = int(k), verbose = False))
        time_est.append(time.time() - t0)
        if verbose:
            print("[%-25s] %d%%" % ('='*(int((i+1)/len(kv)*25)-1) + '>', (i+1)/(len(kv))*100), end = '\r')

        
    # compute the exact value of Z
    t0 = time.time()
    Zexact = ComputeZexact(Φ, indeces)
    time_exact = time.time() - t0
    
    # build the dataframe with the errors
    df = pd.DataFrame(columns = ['i', 'err', 'k'])
    df.i = np.concatenate([indeces for k in kv])
    df.k = np.concatenate([[int(k) for i in range(len(Zexact))] for k in kv])
    df.err = np.concatenate(np.array([np.abs(Z - Zexact)/Zexact for Z in Zest]))
    
    return TestReturn(df, time_est, time_exact)


###############################################################################ààà

def ComputeZest(Φ, indeces, k = 20, Ψ = None, verbose = True, est_method = 'KM'):
    '''
    This function computes the partition function for a set of indeces
    
    Use: ComputeZest(Φ, indeces)
    
    Inputs:
        * Φ (array): embedding to compute the softmax
        * indeces (array): indeces for which the partition function should be computed
        
    Optional inputs:
        * k (int): order of the mixture of Gaussian approximation. By default set to 20
        * Ψ (array): embedding of the output layer, used for the non-symmetric model. By default set to None
        * verbose (bool): if set to True, it will print updates
        * est_method (string): algorithm used to estimate the mixture of Gaussian parameters. It is one value between
            + 'KM': uses K-means (default)
            + 'EM': uses the mixture of Gaussians
        
    Output:
        * Z (array): vector of estimated partition functions (of the samesize of indeces)'''

    
    n, dim = np.shape(Φ)
    
    if Ψ == None: 
        if k > 1:
            if verbose:
                print('Estimating the labels')

            if est_method == 'KM':
                kmeans = faiss.Kmeans(dim, k, verbose = False)
                kmeans.train(np.ascontiguousarray(Φ).astype('float32'))
                _, ℓ = kmeans.assign(np.ascontiguousarray(Φ).astype('float32'))
                
            elif est_method == 'EM':
                gm = GaussianMixture(n_components = k, covariance_type = 'diag').fit(Φ)
                ℓ = gm.predict(Φ)

            else:
                raise DeprecationWarning("The selected method is not valid")

        else:
            ℓ = np.zeros(n)  

        μ = np.array([np.mean(Φ[ℓ == a], axis = 0) for a in range(k)])
        σ2 = np.array([np.var(Φ[ℓ == a], axis = 0) for a in range(k)])
        π = np.array([np.sum(ℓ == a) for a in range(k)])
        
    else:
        
        if Φ.shape != Ψ.shape:
            raise DeprecationWarning("The input and output weight matrices have inconsistent shapes")
        
        if k > 1:
            if verbose:
                print("Estimating the labels")

            if method == 'KM':
                kmeans = faiss.Kmeans(dim, k, verbose = False)
                kmeans.train(np.ascontiguousarray(Ψ).astype('float32'))
                _, ℓ = kmeans.assign(np.ascontiguousarray(Ψ).astype('float32'))

            elif method == 'GM':
                gm = GaussianMixture(n_components = k, covariance_type = 'diag').fit(Ψ)
                ℓ = gm.predict(Ψ)

            else:
                raise DeprecationWarning("The selected method is not valid")

        else:
            ℓ = np.zeros(n)

        μ = np.array([np.mean(Ψ[ℓ == a], axis = 0) for a in range(k)])
        σ2 = np.array([np.var(Ψ[ℓ == a], axis = 0) for a in range(k)])
        π = np.array([np.sum(ℓ == a) for a in range(k)])

    if verbose:
        print("Computing the Z values")

    return np.exp(Φ[indeces]@μ.T + 0.5*Φ[indeces]**2@σ2.T)@π


def ComputeZexact(Φ, indeces, Ψ = None):
    '''This function computes the exact normalization constant given the embedding
    
    Use: Z = ComputeZexact(Φ, indeces)
    
    Inputs:
        * Φ (array): input embedding matrix
        * indeces (array): indeces with respect to whom the cost Z will be computed
    
    Optional inputs:
        * Ψ (array): output weights, used for the non symmetric version of the algorithm
        
    '''

    # split the indeces vector into n chunks of size approximately 1000
    n = np.max([1,int(len(indeces)/1000)])
    
    k, m = divmod(len(indeces), n)
    indeces_split = [indeces[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

    Z = []
    for idx in indeces_split:
        
        if Ψ == None:
            Z.append(np.sum(np.exp(Φ[idx]@Φ.T), axis = 1))
        else:
            Z.append(np.sum(np.exp(Φ[idx]@Ψ.T), axis = 1))
            
    return np.concatenate(Z)