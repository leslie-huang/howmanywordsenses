# How Many Word Senses

Leslie Huang (lesliehuang@nyu.edu)
Kyunghyun Cho
Sam Bowman

This repository contains code for unsupervised estimation of the number of word senses from word contexts extracted from pretrained BERT, ELMo, and fastText models.

# ⚠️ Warning: this code is (at best) in alpha mode. ⚠️

Because our experiments yielded a null result, we have no plans to make this code "camera ready."


For a demo, see the `Sentence_labeled_example.ipynb`, which demonstrates how to extract contexts, compute the vector representations, and visualize the results.

## Contents

- All the main functionality is in the `.py` scripts located in the root of this directory. With our apologies, the documentation is incomplete.

- `Sentence_labeled_example.ipynb` is a walk through with a manually labeled example.

- `/sentence_labeled_example` includes files for the example of contexts containing the word "sentence" with the manual context labels assigned to each context, and the vectors from each model saved as pickles. These files are loaded in the `Sentence_labeled_example.ipynb` notebook.

- `/experiments_code` includes (some of the) files for running the experiments described in our report: specifically, the Bayesian and frequentist experiments with the ELMo model. We also include the top 1,000 words and the script to generated this list.
