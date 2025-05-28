  <font size="7"><span style = "color:#4a5759">EDRep</span></font><br /> 
 <font size="5"><span style = "color:#6b9080"> <b>E</b>fficient <b>d</b>istributed <b>rep</b>resentations with linear-time SoftMax normalization</span></font><br />



| **[Documentation]()** | **[Paper](https://openreview.net/pdf?id=9M4NKMZOPu)** |


This is the Python implementation of the work presented in [(Dall'Amico, Belliardo - *Learning distributed representations with efficient SoftMax normalization*)](https://openreview.net/pdf?id=9M4NKMZOPu). We propose a *2Vec*-like algorithm, formulated for as a general purpose embedding problem. We consider a set of $n$ objects for which we want to obtain a distributed representation $X \in \mathbb{R}^{n\times d}$ in a $d$-dimensional Euclidean space. The algorithm requires a probability matrix $P \in \mathbb{R}^{n\times n}$ as input whose entries $P_{ij}$ are a measure of affinity between the objects $i$ and $j$. We then train the following loss function to generate embedding vectors (contained in the rows of $X$) that best approximate the input distribution described by the matrix $P$.
$$
X = {\rm arg~min}_{Y \in \mathcal{U}_{n\times d}} \sum_{i = 1}^n\Bigg[ - \underbrace{\sum_{j = 1}^n P_{ij}{\rm log}\left({\rm SoftMax}(YY^T)_{ij}\right)}_{\rm cross-entropy} + \underbrace{\frac{1}{n} y_i^T\sum_{j = 1}^n y_j}_{\rm regularization}\Bigg]\,\,,
$$

where $y_i$ is the $i$-th row of $Y$ and $\mathcal{U}_{n \times d}$ is the set of all matrices of size $n\times d$ whose rows have unitary norm. In words, we propose a variational approach based on the use of the softmax function of $XX^T$ to learn the embedding, in the spirit of *2Vec* algorithms. 

In (Dall'Amico, Belliardo - *Learning distributed representations with efficient SoftMax normalization*) we described an efficient way to estimate in $\mathcal{O}(n)$ operations the normalization constants of ${\rm SoftMax}(XX^T)$ and hence to optimize the efficiently the proposed loss function.

In our implementation, we also consider the case in which $P$ is a rectangular matrix $P\in \mathbb{R}^{n\times m}$ and two embedding matrices need to be learned: $X \in \mathbb{R}^{n\times d}, Y \in \mathbb{R}^{m\times d}$. We refer the reader to the paper and to the documentation for further details on this use case.

> We refer the user to the [**paper**](https://openreview.net/pdf?id=9M4NKMZOPu) and to the [**documentation**]() for further details and examples.


## Installation

To install the latest version of `EDRep`, you can run the following command from the terminal.

```bash
pip install git+https://github.com/lorenzodallamico/EDRep.git
```

We also shared an `Anaconda` environment in which all codes were run and tested. You can create it by running the following commands in the terminal

```bash
conda env create -f EDRep_env.yml
conda activate EDRep
```

## Citation

If you make use of these codes, please use the following citation:


```
@misc{dallamico2024efficient,
      title={Efficient distributed representations with linear-time attention scores normalization}, 
      author={Lorenzo Dall'Amico and Enrico Maria Belliardo},
      year={2024},
      eprint={2303.17475},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2303.17475}, 
}
```

## Authors

* [Lorenzo Dall'Amico](https://lorenzodallamico.github.io/) - lorenzo.dallamico@isi.it
* Enrico Maria Belliardo - enrico.m.belliardo@gmail.com

## License
This software is released under the GNU AFFERO GENERAL PUBLIC LICENSE (see included file LICENSE)
