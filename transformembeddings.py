#!/usr/bin/env python3

import fastText
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import spacy
import sys
import tempfile

from allennlp.commands.elmo import ElmoEmbedder
from collections import Counter
from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from itertools import chain, permutations
from umap import UMAP
from scipy.stats import mode
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from subprocess import check_call

from contexts import Contexts, LabeledContexts
from transform_utils import *


class ContextEmbeddings:
    """
    Takes a Contexts or LabeledContexts object, applies processing steps (e.g. masking),
    extracts embeddings from the desired model,
    and decomposes, clusters, and visualizes the data.
    """

    def __init__(
        self,
        ContextsObj,
        tf_idf_weighting,
        mask_pct,
        exclude_word,
        drop_punct=False,
        bert_cls=False,
    ):
        self.word = ContextsObj.word
        self.num_contexts = ContextsObj.num_contexts
        self.contexts = ContextsObj.contexts  # the full contexts
        self.context_type = ContextsObj.context_type
        self.window_size = ContextsObj.window_size

        # processing the contexts
        self.spacy_contexts = None  # list of spacy-fied contexts
        self.exclude_word = exclude_word
        self.drop_punct = drop_punct
        self.mask_pct = (
            mask_pct
        )  # if not None, spacy_contexts have randomly masked words
        self.tf_idf_weighting = tf_idf_weighting
        self.tf_idf_dicts = None

        self.prepare_contexts()

        # extracting the embeddings
        self.embedding_type = None
        self.bert_path = None
        self.bert_cls = bert_cls
        self.ft_path = None
        self.embedding_representation = None
        self.num_dims = None

        self.true_labels = (
            ContextsObj.true_labels
            if isinstance(ContextsObj, LabeledContexts)
            else None
        )

        # decomposing and clustering the embeddings
        self.decomp_method = None
        self.decomp_dims = None
        self.decomposed_embedding_representation = None
        self.cluster_method = None
        self.num_clusters = None
        self.predicted_labels = None

    ##############################
    # Processing the contexts
    ##############################
    def random_mask(self):
        """
        randomly drop (mask) a % of words while preserving order
        from already-tokenized spacy_contexts
        """
        masked_docs = []

        for doc in self.spacy_contexts:
            indices = range(len(doc))
            num_words_keep = int(len(doc) * (1.0 - self.mask_pct))
            if len(indices) > 0:
                keep_indices = sorted(
                    np.random.choice(indices, num_words_keep, replace=False)
                )
            else:
                keep_indices = indices

            masked_doc = [doc[i] for i in keep_indices]
            masked_docs.append(masked_doc)

        self.spacy_contexts = masked_docs

    def prepare_contexts(self):
        """
        Create spacy contexts and tf-idf dictionaries,
        and apply exclude_word and random masking if necessary.
        """
        nlp = spacy.load("en_core_web_sm")

        with nlp.disable_pipes("tagger", "ner", "parser"):
            self.spacy_contexts = [nlp(context) for context in self.contexts]

        tokenized_contexts = []

        for context in self.spacy_contexts:
            tokenized = [t.text for t in context if not t.is_space]

            if self.exclude_word:
                tokenized = [t for t in tokenized if t not in self.word.strip()]

            if self.drop_punct:
                tokenized = [t.text for t in tokenized if not t.is_punct]

            tokenized_contexts.append(tokenized)

        self.spacy_contexts = tokenized_contexts

        # Additional preprocessing
        if self.mask_pct:
            self.random_mask()  # re-populating self.spacy_contexts with randomly masked contexts

        if self.tf_idf_weighting:
            self.tf_idf_dicts = tf_idf_weight(
                self.spacy_contexts
            )  # populating self.tf_idf_dicts

    ##############################
    # Get vector representations
    ##############################

    def extract_embeddings(
        self,
        embedding_type,
        use_pkl=False,
        pkl_path=None,
        bert_path=None,
        ft_path=None,
        subtract_context=False,
        bert_cls = False
    ):
        """
        This method populates self.embedding_representation with the vector representations
        from one chosen embedding model.
        """
        self.embedding_type = embedding_type
        self.subtract_context = subtract_context
        self.bert_cls = bert_cls

        if use_pkl:
            self.embedding_representation = pd.read_pickle(pkl_path)
            # warning: this does not remember the settings used to process the contexts (e.g. masking)
            # prior to extracting the vector representations
            return

        if self.embedding_type == "fastText":
            self.ft_path = ft_path
            self.get_ft_representations()

        if self.embedding_type == "BERT":
            self.bert_path = bert_path
            self.get_bert_representations()

        if self.embedding_type == "ELMO":
            self.get_elmo_representations()

    def get_ft_representations(self):
        sentence_vecs = []

        ft_model = fastText.FastText.load_model(self.ft_path)

        for context_idx, tokenized_context in enumerate(self.spacy_contexts):
            sentence_vec = compute_ft_tokenized_rep(
                tokenized_context,
                self.word,
                ft_model,
                self.tf_idf_weighting,
                self.tf_idf_dicts,
                context_idx,
                self.subtract_context,
            )

            if len(sentence_vec) != 0 and float("-inf") not in sentence_vec:
                sentence_vecs.append(sentence_vec)

        vec_df = pd.DataFrame(np.row_stack(sentence_vecs))

        self.num_dims = vec_df.shape[1]
        self.embedding_representation = vec_df
        self.num_contexts = vec_df.shape[0]

    def get_elmo_representations(self):
        elmo = ElmoEmbedder()
        sentence_vecs = []

        # for each context, for each word, get the ELMO representation
        # then average the hidden layers into a 1xD dimensional vector
        for context_idx, tokenized_context in enumerate(self.spacy_contexts):
            raw_elmo_rep = elmo.embed_sentence(tokenized_context)
            # dimension is 3 x num_tokens x 1024

            sentence_vec = compute_elmo_tokenized_rep(
                tokenized_context,
                self.word,
                raw_elmo_rep,
                self.tf_idf_weighting,
                self.tf_idf_dicts,
                context_idx,
                self.subtract_context,
            )

            if len(sentence_vec) != 0:
                sentence_vecs.append(sentence_vec)

        vec_df = pd.DataFrame(np.row_stack(sentence_vecs))

        self.num_dims = vec_df.shape[1]
        self.embedding_representation = vec_df
        self.num_contexts = vec_df.shape[0]

    def get_bert_representations(self):
        """
        For each sentence:
            For each word, average hidden layer vectors together.
            Then multiply by word's tf-idf weight to get weighted word vector.
            Average all weighted word vectors to compute sentence representation.

        Returns:
            dataframe with each context (as a row) and its vector representation
            (cols = dimensions of the vec representation)
        """
        raw_bert = get_raw_bert_rep(self.bert_path, self.spacy_contexts)
        sentence_vecs = []

        for context_idx, tokenized_context in enumerate(self.spacy_contexts):
            sentence_vec = compute_bert_tokenized_rep(
                tokenized_context,
                self.word,
                raw_bert[context_idx],
                self.tf_idf_weighting,
                self.tf_idf_dicts,
                context_idx,
                self.subtract_context,
                self.bert_cls,
            )
            if sentence_vec is not None and len(sentence_vec) != 0:
                sentence_vecs.append(sentence_vec)

        vec_df = pd.DataFrame(np.row_stack(sentence_vecs))

        self.num_dims = vec_df.shape[1]
        self.embedding_representation = vec_df
        self.num_contexts = vec_df.shape[0]

    def pickle_embeddings(self, filename):
        self.embedding_representation.to_pickle(filename)

    ############################################################
    # Decomposition
    ############################################################

    def decompose_embeddings(self, decomp_method, decomp_dims, additional_params={}):
        """
        Decompose raw word embeddings into lower-dimensional representation.

        @param additional_params {param_name: value} of parameters accepted by the sklearn decomposition function.
            Cannot include n_components.
        """
        self.decomp_method = decomp_method
        self.decomp_dims = decomp_dims

        if self.decomp_method == "PCA":
            pca = PCA(n_components=self.decomp_dims, **additional_params)
            self.decomposed_embedding_representation = pca.fit_transform(
                self.embedding_representation
            )
            self.explained_variance_ratio_ = pca.explained_variance_ratio_

        if self.decomp_method == "TSNE":
            self.decomposed_embedding_representation = TSNE(
                n_components=self.decomp_dims, **additional_params
            ).fit_transform(self.embedding_representation)

        if self.decomp_method == "UMAP":
            self.decomposed_embedding_representation = UMAP(
                n_components=self.decomp_dims, **additional_params
            ).fit_transform(self.embedding_representation)

    ##############################
    # Clustering
    ##############################

    def cluster_embeddings(
        self,
        cluster_method,
        num_clusters,
        use_decomposed,
        additional_params={"random_state": 0},
    ):
        """
        Predicts cluster assignments for data points on raw or decomposed embedding representations.

        @param additional_params {param_name: value} of parameters accepted by the sklearn clustering function.
            Cannot include n_components.
        """
        self.cluster_method = cluster_method
        self.num_clusters = num_clusters

        if use_decomposed:
            vec_df = self.decomposed_embedding_representation
        else:
            vec_df = self.embedding_representation

        if self.cluster_method == "KMeans":
            self.predicted_labels = KMeans(
                n_clusters=self.num_clusters, **additional_params
            ).fit_predict(vec_df)

        if self.cluster_method == "Spectral":
            self.predicted_labels = SpectralClustering(
                n_clusters=self.num_clusters,
                affinity="cosine",
                assign_labels="discretize",
                **additional_params,
            ).fit_predict(vec_df)

        if self.cluster_method == "GaussianMix":
            gm_model = GaussianMixture(
                n_components=self.num_clusters, **additional_params
            ).fit(vec_df)

            self.predicted_labels = gm_model.predict(vec_df)

        if self.cluster_method == "BayesGaussMix":
            bgm_model = BayesianGaussianMixture(
                n_components=self.num_clusters, **additional_params
            ).fit(vec_df)

            self.predicted_labels = bgm_model.predict(vec_df)
            self.num_clusters = len(
                set(self.predicted_labels)
            )  # set num_clusters to actual num components
