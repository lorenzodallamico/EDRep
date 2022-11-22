# P2Vec implementation

This `README` has to be completed and improved


## Content

* `ApproximateSoftmax`: this notebook has still to be done essentially. We need some embeddings and then we need to evaluate time and performance to approximate softmax
* `Community detection`: in this notebook I test the performance of our algorithm on community detection, formulated as an inference problem for which we can measure the accuracy. The performance is tested against my spectral algorithm which is relatively fast and in near Bayes-optimal. The fastest implementation of `Node2Vec` I found (written in `C++`) takes too long to run and thus is not compared.
* `Demo`: this notebook has still to be finished. It will serve as a way to introduce the main tools to the user. For now it just shows some basic plots which may or may not be kept for the final version.
* `Real graph pre-processing`: this notebook elaborates some raw data to generate well formatted input graphs to be used for the tests. It will probably not be shared in the end.
* `Real graphs tests`: performs community detection on **real** graphs and tests the results against `Node2Vec` algorithm. 
* `Text processing`: performs word embeddings and compares the results with the gensim implementation of `word2vec`.
* `data`: this folder contains two files useful to evaluate a word embedding.
* `dataset`: these are the datasets used to perform community detection on real data.
* `embeddings`: in this folder I put the embeddings to test the algorithm, but they will not be uploaded to github.
* `Package`: this folder contains all the relevant source code:

    * `DCSBM`: these functions are used to generate synthetic graphs with a community structure and to run my algorithm for community detection (the one we compare to)
    * `est`: here there are the functions used to approximate softmax (and to that only)
    * `p2vec`: this is the main file in which we have the function to create an embedding of a probability distribution
    * `p2vecNS`: this is the equivalent implementation of our algorithm with negative sampling. It might be useful to make some comparison (shown in the `DEMO` notebook for the moment), but may be dropped in the end.
    * `PWord2vec`: functions used to preprocess text and generate the P matrix for word2vec.


## Things to do

- [ ] Download embeddings (obtained by different algorithms would be great) and test our algorithm against possible competitors.
- [ ] Agree on all plots and run systematic simulations.
- [ ] Find a couple more text datasets to test our method. Inspect the quality of the embedding.
- [ ] Check the code and simplify it where possible. Take care to see if it is possible to obtain some improvements in terms of complexity.
- [ ] Write a decent documentation.
- [ ] Write a paper.
