import numpy as np

from node_embedding import NodeEmbedding
import dcsbm


# set the parameters

n, c = 1000, 10

θ = np.ones(n)
label = np.zeros(n)
C = np.ones((2,2))*c/2

A, ℓ = dcsbm.adj(C,c, label, θ, True)

dim = 32
res = NodeEmbedding(A, dim)