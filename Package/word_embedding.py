from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix, diags
from joblib import Parallel, delayed
import itertools
from time import time


from eder import *


_flat = lambda x:x**0

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
    all_words = [word2idx[w] for w in all_words]
    counter = Counter(text)
    frequency = np.array([counter[word] for word in all_words])
    
    return text, word2idx, frequency



def CoOccurrenceMatrix(text, distance, n):
    '''This function builds the co-occurency matrix between words within distance

    Use : A = CoOccurrenceMatrix(text, distance, n)

    Input:
        * text (list): input text
        * distance (int): distance of the target co-occurent word. It must be positive
        * n (int): total number of words

    Output:
        * A (scipy sparse matrix): co-occurency matrix of size n x n.
    
    '''
    
    if distance == 0:
        raise DeprecationWarning('The distance must be larger than zero')
    
    A_list = []

    for i in range(1, distance+1):
        v = [tuple([a, b]) for a, b in zip(text[distance:], text[:-distance])]
        counter = Counter(v)
        idx1 = [a[0] for a in counter.keys()]
        idx2 = [a[1] for a in counter.keys()]
        A_list.append(csr_matrix((list(counter.values()), (idx1, idx2)), shape = (n,n)))

    A = np.sum(A_list)
    A = A + A.T

    return A

def WordEmbedding(text, dim = 128, f_func = _flat, sparsify = 100, n_epochs = 8, window_size = 1, min_count = 5, verbose = True, 
                k = 1, cov_type = 'full', γ = 0.75, η = 0.85, n_jobs = 8):
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
        * γ (float): negative sampling parameter
        * n_jobs (int): number of parallel jobs used to build the co-occurency matrix

    Outputs:
        * X (array): embedding matrix
        * word2idx (dictionary): mapping between words and embedding indices
    '''

    t0 = time()
    if verbose:
        print('Text pre-processing')
        
       
    text = list(itertools.chain(*text))
    text, word2idx, frequency = PreProcessText(text, min_count)
    n = max(text)+1


    if verbose:
        print('Get the probability matrix')
    

    # split the text
    l = int(len(text)/n_jobs)
    tt = [text[i*l:(i+1)*l] for i in range(n_jobs)]

    if n_jobs > 1:
        if verbose:
            Pl = Parallel(n_jobs = n_jobs, verbose = 8)
        else:
            Pl = Parallel(n_jobs = n_jobs, verbose = 0)

        result = Pl(delayed(CoOccurrenceMatrix)(t, window_size, n) for t in tt)

    else:
        result = [CoOccurrenceMatrix(text, window_size, n)]

    
    f = f_func(frequency)

    # apply a threshold for the decay
    th = np.sort(f)[int(0.95*n)]
    f[f > th] = th*np.sqrt(np.log(f[f > th])/np.log(th))

    # normalize
    f = f/np.mean(f)

    A = np.sum(result)
    
    # get the sparified matrix P
    if sparsify < n:
        if verbose:
            print('Sparsifying the matrix')
    
        idx2 = top_n_idx_sparse(A, sparsify)
        idx1 = np.concatenate([np.ones(len(a))*i for i, a in enumerate(idx2)])
        idx2 = np.concatenate(idx2)
        
        v = np.array(A[(idx1, idx2)])[0]
        A = csr_matrix((v, (idx1, idx2)), shape = (n,n))
        P = diags((A@np.ones(n))**(-1)).dot(A)
            
    tf = time() - t0
    
    # Eder
    if verbose:
        print('Time elapsed before optimization: ' + str(tf))
        print('Computing the embedding')
    X = CreateEmbedding([P], f = f, dim = dim, p0 = frequency**γ, n_epochs = n_epochs, n_prod = 1., sum_partials = False,
                      k = k, verbose = verbose, cov_type = cov_type, η = η)

    return X, word2idx


def top_n_idx_sparse(matrix, th):
    """Return index of top fraction th of top values in each row of a sparse matrix."""
    top_n_idx = []
    for le, ri in zip(matrix.indptr[:-1], matrix.indptr[1:]):
        # n_row_pick = int(th*(ri - le)+1)
        n_row_pick = min(ri - le, th)
        top_n_idx.append(
            matrix.indices[
                le + np.argpartition(matrix.data[le:ri], -n_row_pick)[-n_row_pick:]
            ]
        )
    return top_n_idx