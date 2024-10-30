# EDRep: Efficient Distributed Representations


This is the code related to (Dall'Amico, Belliardo *Efficient distributed representations with linear-time attention scores normalization*). If you use this code please cite the related article. In this paper we show an efficient method to obtain distributed representations of complex entities given a sampling probability matrix encoding affinity between the items. The main result of the article describes an algorithm with linear-in-size complexity to compute the *Softmax* normalization constant, hence avoiding the need to deploy negative sampling to approximate it.


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

## Content

* `Community detection`: this notebook is meant to be provide an example of usage of our algorithm, applied to the problem of community detection described in the paper.
* `utils`: this folder contains all the relevant source code:
    * `dcsbm`: these functions are used to generate synthetic graphs with a community structure and to run the competing community detection algorithms.
    * `EDRep`: this is the main file in which we have the function to create an embedding of a probability distribution
    * `node_embedding`: this is the main function to create a graph node embedding
* `EDRep_env.yml`: this file can be used to create a conda environment with all the useful packages to run our codes.

## Requirements

To easily run our codes you can create a conda environment using the commands

```
conda env create -f EDRep_env.yml
conda activate EDRep
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

## Authors

* [Lorenzo Dall'Amico](https://lorenzodallamico.github.io/) - lorenzo.dallamico@isi.it
* Enrico Maria Belliardo - enrico.belliardo@isi.it

## License
This software is released under the GNU AFFERO GENERAL PUBLIC LICENSE (see included file LICENSE)
