from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix, diags
from joblib import Parallel, delayed
import itertools


from eder import *


def PreProcessText(text, min_count):
    '''This function makes an initial cleaning of the text, dropping occurrencies of most frequent and very unfrequent words.
    The text is then mapped to integers
    
    Use: text, word2idx = PreProcessTex(text, min_count)
    
    Input:
        * text (list): input text
        * min_count (int): minimal number of occurencies needed for a word to be included
        
    Output:
        * text (list): cleaned text
    '''

    # count the words
    counter = Counter(text)

    # all words in the text and their frequency
    all_words = np.array(list(counter.keys()))
    counts = np.array(list(counter.values()))

    words_to_drop = all_words[counts < min_count ]
    total_counts = sum(counts)
    p_drop = dict(zip(all_words, 1- np.sqrt(1e-5*total_counts/counts)))
    for word in words_to_drop:
        p_drop[word] = 1
    rv = np.random.uniform(0,1,len(text))
    
    # drop words
    text = [t for r, t in zip(rv, text) if p_drop[t] < r ]
    
    # recompute the mapping
    counter = Counter(text)
    all_words = np.array(list(counter.keys()))
    word2idx = dict(zip(all_words, np.arange(len(all_words))))

    # convert to integers
    text = [word2idx[t] for t in text]
    

    return text, word2idx



def CoOccurencyMatrix(text, distance, n):
    '''This function builds the co-occurency matrix between words at a given distance

    Use : A = CoOccurencyMatrix(text, distance, n)

    Input:
        * text (list): input text
        * distance (int): distance of the target co-occurent word. It must be positive
        * n (int): total number of words

    Output:
        * A (scipy sparse matrix): co-occurency matrix of size n x n.
    
    '''
    
    if distance == 0:
        raise DeprecationWarning('The distance must be larger than zero')

    v = [tuple([a, b]) for a, b in zip(text[distance:], text[:-distance])]
    counter = Counter(v)
    idx1 = [a[0] for a in counter.keys()]
    idx2 = [a[1] for a in counter.keys()]
    A = csr_matrix((list(counter.values()), (idx1, idx2)), shape = (n,n))

    A + A.T

    return A + A.T

def WordEmbedding(text, dim = 128, f_func = _flat, n_epochs = 8, window_size = 5, min_count = 5, th = 1, verbose = True, 
                k = 1, cov_type = 'diag', η = 0.5, n_jobs = 8):
    '''This function creates a word embedding given a text
    
    Use: X, word2idx = WordEmbedding(text, dim = 128, n_epochs = 8, window_size = 5, min_count = 5, th = 1, verbose = True, 
                k = 1, cov_type = 'diag', η0 = 0.5, n_jobs = 8)
                

    Inputs
        * text (list of lists of strings): input text

    Optional inputs:
        * dim (int): embedding dimensionality. By default set to 128
        * f_func (function): the norm of the word i is f_func(d_i), where d_i is its frequency
        * n_epochs (int): number of training epochs. By default set to 8
        * window_size (int): window size parameter of the Skip-Gram algorithm
        * min_count (int): minimal required number of occurrencies of a word in a text. By default set to 5
        * th (int): all entries in the co-occurency matrix less of equal to th are discarded. By default set to 1
        * verbose (bool): sets the level of verbosity. By default set to True
        * k (int): order of the mixture of Gaussians approximation
        * cov_type (string): determines the covariance type used in the mixture of Gaussians approximation. By default seto to 'diag'
        * η (float): learning parameter. By default set to 0.5
        * n_jobs (int): number of parallel jobs used to build the co-occurency matrix

    Outputs:
        * X (array): embedding matrix
        * word2idx (dictionary): mapping between words and embedding indices
    '''

    if verbose:
        print('Text pre-processing')
        
       
    text = list(itertools.chain(*text))
    text, word2idx = PreProcessText(text, min_count)
    n = max(text)+1

    if verbose:
        print('Get the probability matrix')
    

    # compute the co-occurency matrices
    if n_jobs > 1:
        if verbose:
            Pl = Parallel(n_jobs = n_jobs, verbose = 8)
        else:
            Pl = Parallel(n_jobs = n_jobs, verbose = 0)

        result = Pl(delayed(CoOccurencyMatrix)(text, l, n) for l in range(1, window_size))

    else:
        result = [CoOccurencyMatrix(text, l, n) for l in range(1, window_size)]


    A = np.sum([np.sum(result[:j]) for j in range(len(result))])
    A = A - diags(A.diagonal())
    
    # remove un-frequent entries
    NZ = A.nonzero()
    vals = np.array(A[NZ])[0]
    idx = vals > th
    NZ = [nz[idx] for nz in NZ]
    A = csr_matrix((vals[idx], (NZ[0], NZ[1])), shape = A.shape)

    # compute the probability matrix
    n = A.shape[0]
    d = np.maximum(A@np.ones(n), 1)
    D_1 = diags(d**(-1))
    P = D_1.dot(A)


    f = f_func(d)
    th = np.sort(f)[int(0.95*n)]
    f[f > th] = th
    
    
    # Eder
    if verbose:
        print('Computing the embedding')
    X = CreateEmbedding([P], f = f, dim = dim, n_epochs = n_epochs, n_prod = 1., sum_partials = False,
                      k = k, verbose = verbose, cov_type = cov_type, η = η)

    return X, word2idx


def _flat(x):
    return x