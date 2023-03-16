# EDER: efficient distributed entity representation


This is the code related to (Dall'Amico, Belliardo *Efficient distributed representation of complex entities beyond negative sampling*). If you use this code please cite the article.

```
add ref
```


## Content

* `Test the approximation`: in this notebook we test our approximation on some pre-computed word embeddings. The embeddings are not included but they can be downloaded from here http://vectors.nlpl.eu/repository/. This is the code used to produce Figure 1.
* `Community detection`: in this notebook I test the performance of our algorithm on community detection, formulated as an inference problem for which we can measure the accuracy. This is the code used to produce Figure 2.
* `Real graphs tests`: performs community detection on real graphs and tests the results against `Node2Vec` algorithm. This is the code used to produce Table 1.
* `Text processing`: performs word embeddings and compares the results with the gensim implementation of `word2vec`. This is the code used to obtain the results described in Section 3.2.
* `dataset`: this folder contains the datasets used to perform community detection on real data.
* `Package`: this folder contains all the relevant source code:

    * `dcsbm`: these functions are used to generate synthetic graphs with a community structure and to run the competing community detection algorithms.
    * `eder`: this is the main file in which we have the function to create an embedding of a probability distribution
    * `node_embedding`: this is the main function to create a graph node embedding
    * `word_embedding`: this contains the main functions to create a word embedding


## Authors

* [Lorenzo Dall'Amico](https://lorenzodallamico.github.io/)
* Enrico Maria Belliardo

## License
This software is released under the GNU AFFERO GENERAL PUBLIC LICENSE (see included file LICENSE)
