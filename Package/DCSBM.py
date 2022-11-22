import numpy as np
import itertools
import networkx as nx
from scipy.sparse import csr_matrix, bmat
from sklearn.metrics.cluster import normalized_mutual_info_score
import faiss
import scipy.sparse.linalg
import sys


def adj(C_matrix,c, label, theta):
    ''' Function that generates the adjacency matrix A with n nodes and k communities
    
    Use: A = adj(C_matrix,c, label, theta)
    
    Input:
        * C_matrix (array of size k x k) : affinity matrix of the network C
        * c (scalar) : average connectivity of the network
        * label (array of size n) : vector containing the label of each node
        * theta  (array of size n) : vector with the intrinsic probability connection of each node
    
    Output:
        * A (sparse matrix of size n x n) : symmetric adjacency matrix
    '''

    # number of communities
    k = len(np.unique(label))
    fs = list()
    ss = list()

    n = len(theta)
    c_v = C_matrix[label].T # (k x n) matrix where we store the value of the affinity wrt a given label for each node
    first = np.random.choice(n,int(n*c),p = theta/n) # we choose the nodes that should get connected wp = theta_i/n: the number of times the node appears equals to the number of connection it will have

    for i in range(k): 
        v = theta*c_v[i]
        first_selected = first[label[first] == i] # among the nodes of first, select those with label i
        fs.append(first_selected.tolist())
        second_selected = np.random.choice(n,len(first_selected), p = v/np.sum(v)) # choose the nodes to connect to the first_selected
        ss.append(second_selected.tolist())

    fs = list(itertools.chain(*fs))
    ss = list(itertools.chain(*ss))

    fs = np.array(fs)
    ss  = np.array(ss)

    edge_list = np.column_stack((fs,ss)) # create the edge list from the connection defined earlier

    edge_list = np.unique(edge_list, axis = 0) # remove edges appearing more then once
    edge_list = edge_list[edge_list[:,0] > edge_list[:,1]] # keep only the edges such that A_{ij} = 1 and i > j

    A = csr_matrix((np.ones(len(edge_list[:,0])), (edge_list[:,0], edge_list[:,1])), shape=(n, n))

    return A + A.transpose()


def matrix_C(c_out, c,fluctuation, fraction):
    ''' Function that generates the matrix C

    Use : C_matrix = matrix_C(c_out, c,fluctuation, fraction)
    
    Input:
        * c_out (scalar) : average value of the of diagonal terms
        * c (scalar) : average connectivity of the desired network
        * fluctuation (scalar) : the off diagonal terms will be distributed according to N(c_out, c_out*fluctuation)
        * fraction  (array of size equal to the number of clusters - k -) : vector \pi containing the  fraction of nodes in each class
    
    Output:
        * C_matrix (array of size k x k) : affinity matrix C
        
    '''
    
    n_clusters = len(fraction)
    C_matrix = np.abs(np.random.normal(c_out, c_out*fluctuation, (n_clusters,n_clusters))) # generate the  off diagonal terms
    C_matrix = (C_matrix + C_matrix.T)/2 # symmetrize the  matrix
    nn = np.arange(n_clusters) 
    for i in range(n_clusters):
        x = nn[nn != i]
        C_matrix[i][i] = (c - (C_matrix[:,x]@fraction[x])[i])/fraction[i] # imposing CPi1 = c1

    return C_matrix  


def computeNMI(Φ, ℓ):
    '''This function computes the NMI as inferred from KMeans applied on the embedding Φ
    
    Use: NMI = computeNMI(Φ, ℓ)
    
    Inputs: 
        * Φ (array): embedding from which the labels should be estimated
        * ℓ (array): true labels
        
    Outpus:
        * NMI (float): normalized mutual information score
    '''

    n_clusters = len(np.unique(ℓ))
    kmeans = faiss.Kmeans(np.shape(Φ)[1], n_clusters, verbose = False)
    kmeans.train(np.ascontiguousarray(Φ).astype('float32'))
    _, ℓest = kmeans.assign(np.ascontiguousarray(Φ).astype('float32'))
    
    return normalized_mutual_info_score(ℓest, ℓ)


##################################################################


def find_sol(S, M, r, eps):
    ''' Function that solves Equation 24 through dicotomy
    Use : 
        rp = find_sol(S, M, r)
    Input :
        S (array of size p x p) : diagonal matrix with the smallest eigenvalues of H_r
        M (array os size p x p) : X^T@D@X, where X is the n x p matrix containing the p smallest eigenvectors of H_r
        r (scalar) : value of r for which X and S are computed
    Output :
        rp (scalar) : value of r \in (1, r) solution to Equation 24
    '''
    
    r_small = 1 # r* > r_small
    r_large = r # r* < r_large
    err = 1
    r_old = r_large
    
    while err > eps:
            
        r_new = (r_small + r_large)/2 
        err = np.abs(r_old - r_new)
        
        v = max(np.linalg.eigvalsh(r_new*S + (r-r_new)*M)) # evaluate the largest eigenvalue in the midpoint
        
        if v > (r-r_new)*(1+r*r_new): # update the boundaries
            r_small = r_new
        else:
            
            r_large = r_new
            
        r_old = r_new
            
    return r_large # return the right edge


def find_rho_B(A):
    ''' Function that computes rho(B)
    Use : 
        rho = find_rho_B(A)
    Input :
        A (array of size n x n) : sparse representation of the adjacency matrix
    Output :
        rho (scalar) : leading eigenvalue of the non-backatracking matrix
    '''
    
    n = np.shape(A)[0] # size of the network
    d = np.array(np.sum(A, axis = 0))[0] # degree vector
    D = scipy.sparse.diags(d, offsets = 0) # degree matrix
    I = scipy.sparse.diags(np.ones(n), offsets = 0) # identity matrix
    M = scipy.sparse.bmat([[A, I - D], [I, None]], format='csr') # matrix B'
    vM = scipy.sparse.linalg.eigs(M, k=1, which='LM', return_eigenvectors=False) # find the largest eigenvalue of B'
    return max(vM.real)


def find_zeta(A, rho, n_clusters, eps, verbose):
    ''' Function that calculates the vector zeta on a connected network A given k as zeta_p = min_{r > 1} {r : s_p(H_r) = 0}
    Use : 
        zeta_v, Y = find_zeta(A, rho, n_clusters, eps)
    Input :
        A (sparse matrix of size n) : adjacency matrix of the network
        rho (scalar) : spectral radius of the non-backtracking matrix
        n_clusters (scalar) : number of clusters k
        eps (scalar) : precision of the estimate
    Output :
        zeta_v (array of size k) : vector containing the values of zeta_p for 1 \leq p \leq k
        Y (array of size n x k) : matrix containing the informative eigenvectors on which k-means whould be performed
    '''
    

    d = np.array(np.sum(A, axis = 0))[0] # degree vector
    n = len(d) # size of the network
    D = scipy.sparse.diags(d, offsets = 0) # degree matrix
    I = scipy.sparse.diags(np.ones(n), offsets = 0) # identity matrix
    zeta_v = np.ones(n_clusters)
    Y = np.zeros((n, n_clusters))
    r = np.sqrt(rho) # initialization of r = sqrt{rho(B)}
    i = n_clusters
    
    while i > 1:
        
        delta = 1
        if verbose:
            OUT = 'Estimating zeta : ' + str(i).zfill(2)
            sys.stdout.write('\r%s' % OUT)
            
        while delta > eps: # iterate while r*-r is smaller than eps
        
            H = (r**2-1)*I + D - r*A # Bethe-Hessian
            v, X = scipy.sparse.linalg.eigsh(H, k = i, which = 'SA') # compute the i+1 smallest eigenvalues and eigenvectors
            idx = v.argsort()
            v = v[idx]
            X = X[:,idx]
            S = np.diag(v) 
            M = X.T@D@X
            rp = find_sol(S, M, r, eps) # iterative solution of Equation 24
            delta = np.abs(r - rp) # updated value of delta
            r = rp # r <- r*      
                   
        degeneracy = sum(np.abs(v[1:]-v[-1]) < eps) # calculate the degeneracy of the i-th smallest eigenvalue
        zeta_v[i-degeneracy:i] = r # store the last value of r* found
        Y[:,i-degeneracy:i] += X[:,i-degeneracy:i] # store the corresponding eigenvectors
        i = i-degeneracy
        
        
    return zeta_v, Y


def community_detection(A, *args, **kwargs):
    '''Function to perform community detection on a graph with n nodes and k communities according to Algorithm 2
    
    Use : 
        cluster = community_detection(A, **kwargs)
    Input :
        A (sparse matrix n x n) : adjacency matrix
        **kwargs:
            n_max (scalar) : maximal number of possible classes to look for during the estimation. If not specified set equal to 80
            real_classes (array of size n) : vector containing the true labels of the network. If not specified set to None
            n_clusters (scalar) : number of clusters k. If not specified it will estimate it
            eps (scalar) : precision rate. If not specified set to machine precision
            projection (True/False) : performs the projection on the unitary hypersphere in dimension k, before the k-means step. If not else specified, set to true
            verbose (True/False): determines if an output is printed
            
    Outout :
        X (array): embedding matrix
        
    '''
    
    n_max = kwargs.get('n_max', 80)
    real_classes = kwargs.get('real_classes', [None])
    n_clusters = kwargs.get('n_clusters', None)
    eps = kwargs.get('eps', np.finfo(float).eps)
    projection = kwargs.get('projection', True)
    verbose = kwargs.get('verbose', True)
    
    d = np.array(np.sum(A,axis = 0))[0] # degree vector
    n = len(d) # size of the network
    rho = find_rho_B(A) # r = rho(B)
    
    if n_clusters == None: # it the number of clusters is not known, we estimate it  
        
        n_clusters = 1 
        D_rho_05 = scipy.sparse.diags((d + (rho -1)*np.ones(n))**(-1/2), offsets = 0) 
        L_rho = D_rho_05.dot(A).dot(D_rho_05) # symmetric reduced Laplacian at tau = rho(B)-1
        flag = 0
        while flag == 0:
            if n_clusters < n_max: # the algo will not find more than n_max clusters
                vrho = scipy.sparse.linalg.eigsh(L_rho, k = n_clusters + 1 , which='LA', return_eigenvectors=False) # largest eigenvalues of L_tau
                if min(vrho)> 1/np.sqrt(rho) + np.finfo(float).eps: #  if informative
                    n_clusters += 1
                    if verbose:
                        OUT = 'Number of clusters detected : ' + str(n_clusters)
                        sys.stdout.write('\r%s' % OUT)
                        sys.stdout.flush()
                else:
                    flag = 1
            else:
                flag = 1


    if verbose:
        print('\n')
        
    # find the zeta vector and  the corresponding informative  matrix
    zeta_p, X = find_zeta(A, rho, n_clusters, eps, verbose) 
    if projection == True:
    
        for i in range(n):
            X[i] = X[i]/np.sqrt(np.sum(X[i]**2)) # normalize the rows  of X
    
    return X




