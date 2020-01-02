import numpy as np
import tempfile

from allennlp.commands.elmo import ElmoEmbedder
from collections import Counter
from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from sklearn.decomposition import PCA
from subprocess import check_call


def tf_idf_weight(spacy_contexts):
    docs_dict = Dictionary(spacy_contexts)
    docs_dict.compactify()

    docs_corpus = [docs_dict.doc2bow(doc) for doc in spacy_contexts]

    model_tfidf = TfidfModel(docs_corpus, id2word=docs_dict)
    docs_tfidf = model_tfidf[docs_corpus]

    # Now generate a list of dicts with k,v = "word": tfidf_frequency
    # each dict contains words from one document (sentence)
    doc_tfidf_dicts = []

    for doc in docs_tfidf:
        d = dict()
        for term, freq in doc:
            d[docs_dict[term]] = freq

        doc_tfidf_dicts.append(d)

    return doc_tfidf_dicts


def get_top_words(ce_object, num_words=100):
    """
    Returns the most frequent words across the entire ContextEmbedding.
    Note that random_mask and exclude_word are applied BEFORE this.
    """
    all_flattened = list(chain.from_iterable(ce_object.spacy_contexts))

    all_counter = Counter(all_flattened)

    if num_words > len(all_counter):
        print(
            f"{num_words} requested, but there are only {len(aggregated_counters)} words."
        )
        results = dict(all_counter.most_common(len(all_counter)))
    else:
        results = dict(all_counter.most_common(num_words))

    return results


def find_PCA_threshold(
    embedding_representation, additional_params={}, pct_variance_explained=0.99
):
    """
    @param additional_params Additional parameters to pass to sklearn PCA function. Cannot include n_components.
    """
    full_model = PCA(**additional_params)
    full_model.fit(embedding_representation)
    var_proportion = full_model.explained_variance_ratio_

    while np.sum(var_proportion) >= pct_variance_explained:
        var_proportion = var_proportion[:-1]

    print(
        f"{pct_variance_explained} of variance is explained with {len(var_proportion)} principal components."
    )
    print(f"Fitting PCA with {len(var_proportion)} components now.")

    return len(var_proportion)


#######################################################
#######################################################
# Helper funcs for get_embedding_representation funcs
#######################################################
#######################################################


def compute_subtract_context(
    subtract_context, target_rep, context_token_reps_excl_target
):

    # sum over context minus target word
    sum_context_without_target = np.sum(context_token_reps_excl_target, axis=0)

    if subtract_context:
        averaged_context_without_target = sum_context_without_target / len(
            sum_context_without_target
        )

        sentence_vec = np.subtract(target_rep, sum_context_without_target)

    else:
        sentence_vec = np.add(target_rep, sum_context_without_target)
        sentence_vec /= len(sum_context_without_target) + 1

    return sentence_vec


def get_ft_token_rep(token, ft_model, tf_idf_weighting, tf_idf_dicts, i):
    """
    Get a word's fastText representation, and optionally weight it using tf-idf.
    """
    raw_word_vec = ft_model.get_word_vector(token)

    # divide word_vec by L2 norm, per fastText implementation of get_sentence_vector
    normalized_vec = raw_word_vec / np.linalg.norm(raw_word_vec)

    # multiply the entire context vector by the word's tf-idf weight from the corresponding dict
    # weight = 1 if word not found
    if tf_idf_weighting:
        normalized_vec *= tf_idf_dicts[i].get(token, 1)

    return normalized_vec


def compute_ft_tokenized_rep(
    tokenized_context,
    target_word,
    ft_model,
    tf_idf_weighting,
    tf_idf_dicts,
    i,
    subtract_context,
):
    """
    Manually get_sentence_vector (with optional tf-idf weighting):
    get_word_vector for each word, divide the word_vec by its L2 norm
    multiply the normalized word_vec by the word's tf-idf weight from doc_tfidf_dicts
    sentence_vec is the mean of the normalized, tf-idf weighted word vectors
    """

    target_rep = None
    context_token_reps_excl_target = []

    for token in tokenized_context:
        if token == target_word:
            target_rep = get_ft_token_rep(
                token, ft_model, tf_idf_weighting, tf_idf_dicts, i
            )

        else:
            context_token_rep = get_ft_token_rep(
                token, ft_model, tf_idf_weighting, tf_idf_dicts, i
            )
            context_token_reps_excl_target.append(context_token_rep)

    sentence_vec = compute_subtract_context(
        subtract_context, target_rep, context_token_reps_excl_target
    )

    return sentence_vec


def get_elmo_token_rep(
    token, token_idx, raw_elmo_rep, tf_idf_weighting, tf_idf_dicts, context_idx
):
    """
    Get a word's ELMo representation, and optionally weight it using tf-idf.
    """
    token_rep = raw_elmo_rep[:, token_idx, :]  # 3x1024

    # average the 3 layers
    averaged_token_rep = np.mean(token_rep, axis=0)  # 1x1024

    if tf_idf_weighting:
        weight = tf_idf_dicts[context_idx].get(token, 1)
        averaged_token_rep = [weight * j for j in averaged_token_rep]

    return averaged_token_rep


def compute_elmo_tokenized_rep(
    tokenized_context,
    target_word,
    raw_elmo_rep,
    tf_idf_weighting,
    tf_idf_dicts,
    context_idx,
    subtract_context,
):
    """
    for each line, extract features and their vector representations (1 per hidden layer)
    for each context, enumerate through the words,
    average the 3 layers,
    optionally multiply by tf-idf weight
    """
    context_token_reps_excl_target = []

    for token_idx, token in enumerate(tokenized_context):
        if token != target_word:
            averaged_token_rep = get_elmo_token_rep(
                token,
                token_idx,
                raw_elmo_rep,
                tf_idf_weighting,
                tf_idf_dicts,
                context_idx,
            )
            context_token_reps_excl_target.append(averaged_token_rep)

        else:
            target_rep = get_elmo_token_rep(
                token,
                token_idx,
                raw_elmo_rep,
                tf_idf_weighting,
                tf_idf_dicts,
                context_idx,
            )

    sentence_vec = compute_subtract_context(
        subtract_context, target_rep, context_token_reps_excl_target
    )

    return sentence_vec


def get_raw_bert_rep(bert_path, spacy_contexts):
    """
    Run pytorch_extract_features.py to get raw BERT representation of contexts.
    """
    # write word contexts to tmp input file
    input_fd, input_fp = tempfile.mkstemp(text=True)
    with open(input_fp, "w") as f:
        for tokenized_context in spacy_contexts:
            f.write(" ".join(tokenized_context) + "\n")

    output_fd, output_fp = tempfile.mkstemp(text=True)

    # call bert model on word contexts
    bert_cmd = [
        "python",
        "pytorch_extract_features.py",
        "--input_file",
        input_fp,
        "--output_file",
        output_fp,
        "--bert_model",
        bert_path,
        "--do_lower_case",
    ]

    check_call(bert_cmd, shell=False)

    # read word contexts from tmp output
    with open(output_fp, "r") as f:
        raw_bert = f.readlines()

    return raw_bert


def get_bert_token_rep(
    token, feature, tf_idf_weighting, tf_idf_dicts, context_idx, bert_cls=False
):

    t_vals = [feature["layers"][j]["values"] for j in range(len(feature["layers"]))]

    token_averaged_layers = [np.mean(x) for x in zip(*t_vals)]

    if bert_cls:
        return token_averaged_layers

    else:
        # multiply averaged word vector by word tf-idf weight
        # weight = 1 if word not found
        if tf_idf_weighting:
            weight = tf_idf_dicts[context_idx].get(token, 1)
            token_averaged_layers = [weight * i for i in token_averaged_layers]

        return token_averaged_layers


def compute_bert_tokenized_rep(
    context,
    target_word,
    raw_bert,
    tf_idf_weighting,
    tf_idf_dicts,
    context_idx,
    subtract_context,
    bert_cls,
):
    if bert_cls:
        excluded_features = ["SEP]"]
    else:
        excluded_features = ["[CLS]", "[SEP]"]

    raw_context_rep = eval(raw_bert)
    features = raw_context_rep["features"]

    context_token_reps_excl_target = []

    target_rep = None

    # for each word, average hidden layers
    for feature in features:
        token = feature["token"]
        # print(token)

        if token not in excluded_features:
            if bert_cls and token == "[CLS]":
                cls_rep = get_bert_token_rep(
                    token,
                    feature,
                    tf_idf_weighting,
                    tf_idf_dicts,
                    context_idx,
                    bert_cls,
                )
            elif token != target_word.lower():  # because BERT is uncased
                token_averaged_layers = get_bert_token_rep(
                    token, feature, tf_idf_weighting, tf_idf_dicts, context_idx
                )
                context_token_reps_excl_target.append(token_averaged_layers)
            else:
                target_rep = get_bert_token_rep(
                    token, feature, tf_idf_weighting, tf_idf_dicts, context_idx
                )

    if target_rep is not None:
        sentence_vec = compute_subtract_context(
            subtract_context, target_rep, context_token_reps_excl_target
        )
        if bert_cls:
            # concat cls token representation to sentence vector
            sentence_vec = np.concatenate((sentence_vec, cls_rep), axis=0)

    else:
        print("No BERT rep for the target word")
        sentence_vec = None

    return sentence_vec
