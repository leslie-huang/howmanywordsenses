# How Many Word Senses

Leslie Huang (lesliehuang@nyu.edu)

Kyunghyun Cho

Sam Bowman

This repository contains code for unsupervised estimation of the number of word senses from word contexts extracted from pretrained BERT, ELMo, and fastText models.

Read about what we did: [link to Medium]

# ⚠️ Warning: this code is (at best) in alpha mode. ⚠️

Because our experiments yielded a null result, we have no plans to make this code "camera ready." With our apologies, the documentation is incomplete.

For a demo, see the `Mouse_labeled_example.ipynb`, which demonstrates how to extract contexts, compute the vector representations, and visualize the results with contexts of the word "mouse."

## Contents

- All the main functionality is in the `.py` scripts located in the root of this directory.
  - Note that `pytorch_extract_features.py` is from https://github.com/huggingface/.


- Software requirements are in `requirements.txt`.

- `Mouse_labeled_example.ipynb` is a walk through with a manually labeled example.

- `/mouse_labeled_example` includes files for the example of contexts containing the word "mouse" with the manual context labels assigned to each context, and the vectors from each model saved as pickles. These files are loaded in the `Mouse_labeled_example.ipynb` notebook.

- `/experiments_code` includes (some of the) files for running the experiments described in our report: specifically, the Bayesian and frequentist experiments with the ELMo model. We also include the top 1,000 words and the script to generated this list.
