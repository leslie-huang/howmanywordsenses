#!/usr/bin/env python3

import json
import numpy as np
import os
import pandas as pd
import re
import spacy
import sys
from nltk import sent_tokenize, word_tokenize
from collections import Counter
from itertools import chain


class Contexts:
    def __init__(self, corpus_db, word, db_table, db_table_col):
        """
        @param corpus_db path to corpus as SQL database
        @param word target word
        @param db_table name of the database table
        @param db_table_col name of database table column containing text
        """
        self.corpus_db = corpus_db
        self.db_table = db_table
        self.db_table_col = db_table_col
        self.corpus_size = pd.read_sql_query(
            f"SELECT COUNT(*) AS c FROM {self.db_table}", self.corpus_db
        )["c"][0]

        # All attributes below are also in LabeledContexts
        self.word = word
        self.randomized = None
        self.context_type = None
        self.window_size = None

        self.contexts = None  # list

        self.num_contexts = None

    ##############################
    # Get contexts
    ##############################

    def select_contexts(self, context_type, window_size, num_contexts, randomized):
        """
        @param context_type sentence or window
        @param window_size if context_type is window, window_size should be a positive nonzero integer
        @param num_contexts (maximum) number of contexts to retrieve
        @param randomized retrieve random contexts from the database

        This method sets all attributes related to selecting raw word contexts from the corpus.
        """
        nlp = spacy.load("en_core_web_sm")

        self.context_type = context_type
        self.window_size = window_size
        self.randomized = randomized

        # select random rows or rows ordered by ID
        randomized_query = "ORDER BY RANDOM()" if self.randomized else "ORDER BY id"

        continue_condition = True
        i = 1

        while continue_condition:
            buffer_count = i * 500

            docs_containing_word = pd.read_sql_query(
                f"SELECT * FROM {self.db_table} \
                    WHERE {self.db_table_col} LIKE '%{self.word}%' \
                    LIMIT {num_contexts + buffer_count}",
                self.corpus_db,
            )

            word_contexts = []

            for utterance in docs_containing_word[self.db_table_col].tolist():
                if self.context_type == "sentence":
                    word_contexts += self.get_sentence_window(
                        nlp, utterance, self.word
                    )  # RHS is a list
                if self.context_type == "window":
                    word_contexts += self.get_window(
                        nlp, utterance, self.word, self.window_size
                    )  # RHS is a list
                if self.context_type == "document":
                    word_contexts.append(utterance)

            i += 1

            # stop when finally have enough contexts, or after a certain number of loops
            if (len(word_contexts) >= num_contexts) or (
                (self.corpus_size - buffer_count) < 500
            ):
                continue_condition = False

        # word may occur 1+ times in an utterance, so "sentence" or "window" options
        # may result in len(word_contexts) > num_examples if all utterances are processed
        # enforce num_contexts as the max
        limit = min(len(word_contexts), num_contexts)
        self.contexts = word_contexts[:limit]
        self.num_contexts = limit

    def get_sentence_window(self, nlp, utterance, word):
        """
        @param nlp Spacy NLP
        @param utterance document as str
        @param word target word

        Returns list of sentences (str) that contain the word
        Returns [] if word not found properly in sentence
        """
        doc_sentences = [sent for sent in sent_tokenize(utterance)]

        sentence_contexts = []

        for sent in doc_sentences:
            with nlp.disable_pipes("tagger", "ner", "parser"):
                spacy_doc = nlp(sent)

                word_tokenized_sentence = [token.text for token in spacy_doc]
                if word in word_tokenized_sentence:
                    sentence_contexts.append(sent)

        return list(set(sentence_contexts))

    def get_window(self, nlp, utterance, word, window_size):
        """
        @param utterance document as str
        @param word target word
        @param window_size window +/- word to capture

        Returns List of str (word in context). If word occurs multiple times in text,
        all occurrences will be in list.
        Returns [] if word not found properly in sentence
        """
        with nlp.disable_pipes("tagger", "ner", "parser"):
            spacy_doc = nlp(utterance)

        text_tokens = [token.text for token in spacy_doc]
        target_word_indices = [
            i for i, elem in enumerate(text_tokens) if elem.strip() == word
        ]

        windows = []

        for i in target_word_indices:
            left_index = max(0, i - window_size)
            right_index = min(len(text_tokens), i + window_size) + 1

            windows.append(" ".join(text_tokens[left_index:right_index]))

        return list(set(windows))

    def save_contexts_json(self, filename):
        """
        @param filename file name ending in .json
        Save the word contexts to a json with columns ["cluster", "context"]
        """
        with open(filename, "w") as f:
            pd.DataFrame({"context": self.contexts, "cluster": ""}).to_json(
                f, orient="records"
            )


class LabeledContexts:
    """
    Class where manually labeled contexts are loaded.
    """

    def __init__(self, file_path, context_data):
        """
        @param file_path path to labeled context data
        @param context_data accepts either a Contexts instance or a dict of the necessary attributes
        """

        if isinstance(context_data, Contexts):
            # attributes extracted from Contexts
            self.word = context_data.word
            self.randomized = context_data.randomized
            self.context_type = context_data.context_type
            self.window_size = context_data.window_size

        else:  # assumes context_data is a dict with keys: word, randomized, context_type, window_size
            self.word = context_data["word"]
            self.randomized = context_data["randomized"]
            self.context_type = context_data["context_type"]
            self.window_size = context_data["window_size"]

        labeled_contexts = pd.read_json(file_path, encoding="latin1")
        self.contexts = labeled_contexts["context"].tolist()
        self.true_labels = labeled_contexts["cluster"].tolist()

        self.num_contexts = len(self.true_labels)


############################################
# Utils that handle Contexts objects
############################################
def get_corpus_top_words(ContextsObj, num_words):
    """
    @param ContextsObj Contexts object
    @param num_words number of top words to return

    Returns dictionary of most frequent words (not including punctuation
    or numbers) in a Contexts object
    """
    nlp = spacy.load("en_core_web_sm")

    all_docs = pd.read_sql_query(
        f"SELECT {ContextsObj.db_table_col} FROM {ContextsObj.db_table}",
        ContextsObj.corpus_db,
    )[ContextsObj.db_table_col].tolist()

    all_tokenized = []

    with nlp.disable_pipes("tagger", "ner", "parser"):
        for doc in all_docs:
            tokenized = [t.text for t in nlp(doc) if (not t.is_space)]
            all_tokenized.append(tokenized)

    all_counter = Counter(list(chain.from_iterable(all_tokenized)))

    if num_words > len(all_counter):
        print(
            f"{num_words} requested, but there are only {len(aggregated_counters)} words."
        )
        return all_counter.most_common(len(all_counter))
    else:
        return all_counter.most_common(num_words)
