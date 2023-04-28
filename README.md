# EDRep: Efficient Distributed Representations


This is the code related to (Dall'Amico, Belliardo *Efficient distributed representation beyond negative sampling*). If you use this code please cite the related article. In this paper we show an efficient method to obtain distributed representations of complex entities given a sampling probability matrix encoding affinity between the items.

```
@misc{dallamico2023efficient,
      title={Efficient distributed representations beyond negative sampling}, 
      author={Lorenzo Dall'Amico and Enrico Maria Belliardo},
      year={2023},
      eprint={2303.17475},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


## Content

* `Notebooks`: this folder contain the notebooks needed to reproduce the results present in our paper.
    * `Test the approximation`: in this notebook we test our approximation on some pre-computed word embeddings. The embeddings are not included but they can be downloaded from http://vectors.nlpl.eu/repository/. This is the code used to produce Figure 1.
    * `Community detection`: in this notebook I test the performance of our algorithm on community detection, formulated as an inference problem for which we can measure the accuracy. This is the code used to produce Figure 2.
    * `Real graphs tests`: performs community detection on real graphs and tests the results against `Node2Vec` algorithm. This is the code used to produce Table 1.
    * `Text processing`: performs word embeddings and compares the results with the gensim implementation of `word2vec`. This is the code used to obtain the results described in Section 3.2.
* `dataset`: this folder contains the datasets used to perform community detection on real data.
* `Package`: this folder contains all the relevant source code:

    * `dcsbm`: these functions are used to generate synthetic graphs with a community structure and to run the competing community detection algorithms.
    * `edr`: this is the main file in which we have the function to create an embedding of a probability distribution
    * `node_embedding`: this is the main function to create a graph node embedding
    * `word_embedding`: this contains the main functions to create a word embedding

* `EDRep_env.yml`: this file can be used to create a conda environment with all the useful packages to run our codes.

## Requirements

To easily run our codes you can create a conda environment using the commands

```
conda env create -f EDRep_env.yml
conda activate EDRep
```

On top of this, in order to use the `web.datasets.similarity` package you should follow the installation instructions of https://github.com/kudkudak/word-embeddings-benchmarks. Similarly, the `node2vec` package can be installed following the instructions at https://github.com/thibaudmartinez/node2vec (Linux 64 only). Finally, if the `faiss` package creates problem, you might need to install its dependency manually with

```
sudo apt-get install libstdc++6
```

## Use

For the use of our package, we invite the practitioner to refer to the jupyter notebooks contained in the `Notebooks` folder. We here report the comment lines of the three main functions that can used from our package.

* The function `CreateEmbedding` is the main function and it provides a distributed representation given a probability matrix

```python
X = CreateEmbedding(Pv)

'''
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
```
The function `NodeEmbedding` creates a node distributed representation of a graph given its adajcency matrix representation. The graph can be directed or undirected and it can be weigted, but weights must be non-negative.

```python
'''Algorithm for node embedding using EDRep
    
    * Use: X = NodeEmbedding(A, dim)
    
    * Inputs:
        * A (scipy sparse matrix): graph adjacency matrix. It can be weighted and non-symmetric, but its entries must be non-negative
        * dim (int): embedding dimension
        
    * Optional inputs:
        * f_func (function): the norm of x_i will be set to f(d_i)
        * n_epochs (int): number of training epochs in the optimization. By default set to 35   
        * n_prod (int): maximal distance reached by the random walker. By default set to 1
        * k (int): order of the mixture of Gaussian approximation. By default set to 1
        * cov_type (string): determines the covariance type in the optimization process. Can be 'diag' or 'full'
        * verbose (bool): if True (default) it prints the update
        * η (float): learning rate
        
    * Output:
        * X (array): embedding matrix
    '''
    ```

## Authors

* [Lorenzo Dall'Amico](https://lorenzodallamico.github.io/) - lorenzo.dallamico@isi.it
* Enrico Maria Belliardo - enrico.belliardo@isi.it

## License
This software is released under the GNU AFFERO GENERAL PUBLIC LICENSE (see included file LICENSE)
