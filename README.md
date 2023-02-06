# Eder implementation

This `README` has to be completed and improved


## Content

* `Test the approximation`: in this notebook we test our approximation on some pre-computed word embeddings. The embeddings are not included but they can be downloaded from here http://vectors.nlpl.eu/repository/. The test is run on 4 datasets. Maybe we can consider more.
* `Community detection`: in this notebook I test the performance of our algorithm on community detection, formulated as an inference problem for which we can measure the accuracy. The performance is tested against my spectral algorithm which is relatively fast and in near Bayes-optimal. The fastest implementation of `Node2Vec` I found (written in `C++`) takes too long to run and thus is not compared.
* `Real graphs tests`: performs community detection on **real** graphs and tests the results against `Node2Vec` algorithm. 
* `Text processing`: performs word embeddings and compares the results with the gensim implementation of `word2vec`.
* `data`: this folder contains two files useful to evaluate a word embedding.
* `dataset`: these are the datasets used to perform community detection on real data.
* `embeddings`: in this folder I put the embeddings to test the algorithm, but they will not be uploaded to github.
* `Package`: this folder contains all the relevant source code:

    * `dcsbm`: these functions are used to generate synthetic graphs with a community structure and to run my algorithm for community detection (the one we compare to)
    * `eder`: this is the main file in which we have the function to create an embedding of a probability distribution
    * `node_embedding`: this is the main function to create a graph node embedding
    * `word_embedding`: this contains the main functions to create a word embedding


## Things to do

- [x] Download embeddings (obtained by different algorithms would be great) and test our algorithm against possible competitors.
    > There are essentially no competitors. Maybe we can download some more to make the tests a bit more extensive and to use the graph embeddings of the real graphs as well.
- [ ] Agree on all plots and run systematic simulations.
- [ ] Find a couple more text datasets to test our method. Inspect the quality of the embedding.
- [ ] Check the code and simplify it where possible. Take care to see if it is possible to obtain some improvements in terms of complexity.
- [ ] Write a decent documentation.
- [ ] Write a paper.
- [ ] Sparsify the matrix P for Word2Vec
- [ ] Understand better how to use the scheduler of the learning rate.
- [ ] Parallelize gradient descent
- [ ] Consider different types of normalization (l2, l4, lp, linfty)
- [ ] Consider if going back to AdamW
