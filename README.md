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

On top of this, in order to use the `web.datasets.similarity` package you should follow the installation instructions of https://github.com/kudkudak/word-embeddings-benchmarks. Finally, if the `faiss` package creates problem, you might need to install its dependency manually with

```
sudo apt-get install libstdc++6
```

## Authors

* [Lorenzo Dall'Amico](https://lorenzodallamico.github.io/) - lorenzo.dallamico@isi.it
* Enrico Maria Belliardo - enrico.belliardo@isi.it

## License
This software is released under the GNU AFFERO GENERAL PUBLIC LICENSE (see included file LICENSE)
