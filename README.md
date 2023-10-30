# EDRep: Efficient Distributed Representations


This is the code related to (Dall'Amico, Belliardo *Efficient distributed representation beyond negative sampling*). If you use this code please cite the related article. In this paper we show an efficient method to obtain distributed representations of complex entities given a sampling probability matrix encoding affinity between the items. The main result of the article describes an algorithm with linear-in-size complexity to compute the *Softmax* normalization constant, hence avoiding the need to deploy negative sampling to approximate it.


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
    * `Data_completion`: this notebook makes the test presented in the appendix in which we apply our method to the "asymmetric" setting.

* `dataset`: this folder contains the datasets used to perform community detection on real data.
* `utils`: this folder contains all the relevant source code:

    * `dcsbm`: these functions are used to generate synthetic graphs with a community structure and to run the competing community detection algorithms.
    * `EDRep`: this is the main file in which we have the function to create an embedding of a probability distribution
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

</br>

* The function `CreateEmbedding` is the main function and it provides a distributed representation given a probability matrix

    ```python
    res = CreateEmbedding(Pv)
    ```

    > **Inputs**:
    >
    >>   * Pv (list sparse array): the P matrix is provided by the product of all the elements appearing in Pv.from right to left. If `sum_partials = True` the total matrix `P = (Pv[0] + Pv[1]@Pv[0] + Pv[2]@Pv[1]@Pv[0] + ...)/len(Pv) `. If the resulting matrix P is not a probability matrix (hence it its rows do not sum up to 1), a warning is raised.
    >
    > **Optional inputs**:
    >
    >>   * dim (int): dimension of the embedding. By default set to 128
    >>   * p0 (array): array of size n that specifies the "null model" probability
    >>   * n_epochs (int): number of iterations in the optimization process. By default set to 30
    >>   * sum_partials (bool): refer to the description of `Pv` for the use of this parameter. The default value is `False`
    >>   * k (int): order of the GMM approximation. By default set to 1
    >>   * η (float): largest adimissible learning rate. By default set to 0.8.
    >>   * verbose (bool): determines whether the algorithm produces some output for the updates. By default set to True
    >>   * sym (bool): if True (default) generates a single embedding, while is False it generates two
    >
    >    
    > **Output**: The function returns a class with the following elements:
    >
    >>  * EDRep.X (array): solution to the optimization problem for the input weights
    >>  * EDRep.Y (array): solution to the optimization problem for the input weights
    >>  * EDRep.ℓ (array): label vector

</br>

* The function `NodeEmbedding` creates a node distributed representation of a graph given its adajcency matrix representation. The graph can be directed or undirected and it can be weigted, but weights must be non-negative.

    ```python
    res = NodeEmbedding(A, dim)
    ```

    > **Inputs**:
    >
    >>  * A (scipy sparse matrix): graph adjacency matrix. It can be weighted and non-symmetric, but its entries must be non-negative
    >>  * dim (int): embedding dimension
    >    
    > **Optional inputs**:
    >
    >>  * n_epochs (int): number of training epochs in the optimization. By default set to 30
    >>  * walk_length (int): maximal distance reached by the random walker. By default set to 5
    >>  * k (int): order of the mixture of Gaussian approximation. By default set to 1
    >>  * verbose (bool): if True (default) it prints the update
    >>  * η (float): learning rate, by default set to 0.5
    >>  * sym (bool): determines whether to use the symmetric (detfault) version of the algoritm
    >    
    > **Output**:
    >> * res: EDREp class

</br>


* Finally, the matrix `WordEmbedding` provides a word distributed representation generated from a text.

    ```python
    res, word2idx = WordEmbedding(text)
    ```
    > **Inputs**
    >
    >>  * text (list of lists of strings): input text
    >>  * dim (int): embedding dimensionality
    >
    > **Optional inputs**:
    >
    >>  * n_epochs (int): number of training epochs. By default set to 10
    >>  * window_size (int): window size parameter of the Skip-Gram algorithm
    >>  * min_count (int): minimal required number of occurrencies of a word in a text. By default set to 5
    >>  * verbose (bool): sets the level of verbosity. By default set to True
    >>  * k (int): order of the mixture of Gaussians approximation
    >>  * γ (float): negative sampling parameter
    >>  * η (float): learning parameter. By default set to 0.5
    >>  * n_jobs (int): number of parallel jobs used to build the co-occurency matrix
    >>  * sym (bool): if True (default) the algorithm is run in its symmetric version
    >
    > **Outputs**:
    >>  * res: EDRep class
    >>  * word2idx (dictionary): mapping between words and embedding indices

## Authors

* [Lorenzo Dall'Amico](https://lorenzodallamico.github.io/) - lorenzo.dallamico@isi.it
* Enrico Maria Belliardo - enrico.belliardo@isi.it

## License
This software is released under the GNU AFFERO GENERAL PUBLIC LICENSE (see included file LICENSE)
