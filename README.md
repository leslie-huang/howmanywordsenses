# How Many Word Senses

_Leslie Huang (lesliehuang@nyu.edu)_

_Kyunghyun Cho_

_Sam Bowman_

This repository contains code for unsupervised estimation of the number of word senses from word contexts extracted from pretrained BERT, ELMo, and fastText models.

For a big-picture overview of what we did, [read our post on Medium](https://medium.com/@leslie_huang/automatic-extraction-of-word-senses-from-deep-contextualized-word-embeddings-2f09f16e820).

# ⚠️ Warning: this code is in alpha mode. ⚠️

For a demo, see the `Mouse_labeled_example.ipynb`, which demonstrates how to extract contexts, compute the vector representations, and visualize the results with contexts of the word "mouse."

Code in the modules functions; however, because these experiments yielded a null result, the modules are not "camera ready."

## Contents

- All the main functionality is in the `.py` scripts located in the root of this directory.
  - Credit: `pytorch_extract_features.py` is from [Hugging Face](https://github.com/huggingface/).

- Software requirements are in `requirements.txt`.

- `Mouse_labeled_example.ipynb` is a walk through with a manually labeled example. Note that the notebook currently does not display properly when previewed on GitHub.com.

- `/mouse_labeled_example` includes files for the example of contexts containing the word "mouse" with the manual context labels assigned to each context, and the vectors from each model saved as pickles. These files are loaded in the `Mouse_labeled_example.ipynb` notebook.

- `/experiments_code` includes (some of the) files for running the experiments described in our report: specifically, the Bayesian and frequentist experiments with the ELMo model. We also include the top 1,000 words and the script to generate this list.

## What else you'll need

- A pretrained [BERT model](https://github.com/google-research/bert) (we used `uncased_L-12_H-768_A-12`).
- Our fastText model trained on Wikipedia. (currently unavailable due to file size)
- Our SQLite database of Wikipedia texts (derived from a recent Simple Wikipedia dump): download from [Google Drive](https://drive.google.com/file/d/1mVIFvYiuje5tIjdIkoPoHxDlbBn3daGC/view?usp=sharing)
