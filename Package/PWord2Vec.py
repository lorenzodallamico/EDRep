import numpy as np
from scipy.sparse import csr_matrix, diags
import itertools
from collections import Counter
import multiprocessing as mp
from numba import jit


import warnings
warnings.filterwarnings("ignore")


def PreprocessTextDictionaries(text, th, min_count):
    '''This function preprocesses the text and creates a dictionary mapping each word to an integer and it assigns a probability to be dropped to each word
    
    Use: word2idx, pdrop = PreprocessTextDictionaries(text)
    
    Inputs:
        * text (list of lists of strings): this is the input text in which each sentence corresponds to a list
        * th (float): words will be dropped with a probability 1 - sqrt(th/f), where f is their frequency.
        * min_count (int): all words appearing less then min_count times will be dropped
        
    Outputs:
        * word2idx (dictionary): mapping between words and indeces
        * pdrop (dictionary): mapping between words and probability to be dropped
    '''
    
    # flatten the list
    text_t = list(itertools.chain(*text))
    
    # count the words
    counter = Counter(text_t)

    # free some space
    del text_t
    
    # find the words to drop and delete them from the dictionary
    words_to_exclude = np.array([a for a in counter.keys() if counter[a] < min_count])
    for w in words_to_exclude:
        del counter[w]

    # create the word2idx dictionary
    all_words = [a for a in counter.keys()]
    n = len(all_words)
    word2idx = dict(zip(all_words, np.arange(n)))

    # create the pdrop dictionary
    counts = np.array([a for a in counter.values()])
    total_counts = np.sum(counts)
    pdrop = dict(zip(all_words, 1 - np.sqrt(th*total_counts/counts)))
    for w in words_to_exclude:
        pdrop[w] = 1

    
    return word2idx, pdrop


def build_SG_adjacency_matrix_unit(text, window_size, word2idx, pdrop):
    '''This function generates a weighted adjacency matrix using the skip-gram co-occurences found in the input text
    
    Use: A = build_SG_adjacency_matrix_unit(text, window_size, word2idx, pdrop)
    
    Inputs:
        * text (list of lists of strings): this is the input text in which each sentence corresponds to a list
        * window_size (int): maximal window size used in the skip-gram algorithm
        * word2idx (dictionary): mapping between words and indeces
        * pdrop (dictionary): mapping between words and probability to be dropped
        
    Output:
        * A (sparse array): sparse representation of the weighted adjacency matrix      
    '''
    
    
    n = len(word2idx.keys())
    A = csr_matrix(([0], ([0],[0])), shape = (n,n))

    for i, t in enumerate(text):
        p = np.array([pdrop[a] for a in t])
        r = np.random.uniform(0,1, len(t))
        sentence = [word2idx[a] for i, a in enumerate(t) if pdrop[a] < r[i]]

        idx1, idx2 = GetSkipGrams(sentence, window_size)
        
        WEL = Counter([tuple([a, b]) for a, b in zip(idx1, idx2)])
        idx1 = [a[0] for a in WEL.keys()]
        idx2 = [a[1] for a in WEL.keys()]
        vals = [a for a in WEL.values()]
        
        A += csr_matrix((vals, (idx1, idx2)), shape = (n,n))

    return A


def build_SG_adjacency_matrix(text, window_size, word2idx, pdrop, n_workers):
    '''This function generates a weighted adjacency matrix using the skip-gram co-occurences found in the input text.
    
    Use: A = build_SG_adjacency_matrix(text, window_size, word2idx, pdrop)
    
    Inputs:
        * text (list of lists of strings): this is the input text in which each sentence corresponds to a list
        * window_size (int): maximal window size used in the skip-gram algorithm
        * word2idx (dictionary): mapping between words and indeces
        * pdrop (dictionary): mapping between words and probability to be dropped
        * n_workers (int): number of cores to be used
        
    Output:
        * A (sparse array): sparse representation of the weighted adjacency matrix      
    '''
    
    n_workers = np.min([n_workers, len(text)])
    
    if n_workers == 1:
        A = build_SG_adjacency_matrix_unit(text, window_size, word2idx, pdrop)
        
    else:
        # split the text and run the algorithm in parallel
        positions = [i*int(len(text)/n_workers) for i in range(n_workers+1)]
        positions[-1] = len(text)
        t_split = [text[positions[i]: positions[i+1]] for i in range(n_workers)]

        # run the process in parallel
        pool = mp.Pool(n_workers)
        results = pool.starmap_async(build_SG_adjacency_matrix_unit, [(t, window_size, word2idx, pdrop) for t in t_split])
        pool.close()

        A = np.sum(results.get())

    return A


@jit(nopython = True)
def GetSkipGrams(sentence, window_size):
    '''Given a sentence, this function computes the skip-grams
    
    Use: idx1, idx2 = GetSkipGrams(sentence, window_size)
    
    Inputs:
        * sentence (np array): list of indeces forming the input text.
        * window_size (int): maximal window size to consider a word to be in the context of another
        
    Outputs:
        * idx1 (list): indeces of the central words
        * idx1 (list): indeces of the corresponding context words
        
    '''
    
    idx1 = [0]
    idx2 = [0]
    
    # window size for each central word
    rv = np.random.randint(1, window_size+1, len(sentence)) 
    
    for a in range(len(sentence)):
        
        r = rv[a]
        bmin = max(1, a - r)
        bmax = min(a + r, len(sentence)-1)

        # loop over the context words
        for b in range(bmin,bmax+1):
            if b != a:
                idx1.append(sentence[a])
                idx2.append(sentence[b])
                
    return idx1[1:], idx2[1:]