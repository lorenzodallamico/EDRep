# Eder implementation

This `README` has to be completed and improved


## Content

* `Test the approximation`: in this notebook we test our approximation on some pre-computed word embeddings. The embeddings are not included but they can be downloaded from here http://vectors.nlpl.eu/repository/. The test is run on 4 datasets. Maybe we can consider more.
* `Community detection`: in this notebook I test the performance of our algorithm on community detection, formulated as an inference problem for which we can measure the accuracy. The performance is tested against my spectral algorithm which is relatively fast and in near Bayes-optimal. The fastest implementation of `Node2Vec` I found (written in `C++`) takes too long to run and thus is not compared.
* `Real graphs tests`: performs community detection on **real** graphs and tests the results against `Node2Vec` algorithm. 
* `Text processing`: performs word embeddings and compares the results with the gensim implementation of `word2vec`.
* `dataset`: these are the datasets used to perform community detection on real data.
* `embeddings`: in this folder I put the embeddings to test the algorithm, but they will not be uploaded to github.
* `Package`: this folder contains all the relevant source code:

    * `dcsbm`: these functions are used to generate synthetic graphs with a community structure and to run my algorithm for community detection (the one we compare to)
    * `eder`: this is the main file in which we have the function to create an embedding of a probability distribution
    * `node_embedding`: this is the main function to create a graph node embedding
    * `word_embedding`: this contains the main functions to create a word embedding
