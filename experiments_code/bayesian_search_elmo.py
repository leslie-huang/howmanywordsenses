#!/usr/bin/env python3

import argparse
import contextlib
import copy
import itertools
import os
import pickle
import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from collections import Counter
from multiprocessing import Pool

from contexts import Contexts, LabeledContexts
from transformembeddings import ContextEmbeddings

parser = argparse.ArgumentParser()
parser.add_argument("--word", type=str, help="word")
parser.add_argument("--embedding_type", type=str, help="embedding type")

args = parser.parse_args()

## PATHS
target_save_dir = "/home/lh1036"
if args.word.islower():
    output_fn = (
        f"{target_save_dir}/wiki_bgm_elmo/{args.word}_{args.embedding_type}.json"
    )
else:
    output_fn = (
        f"{target_save_dir}/wiki_bgm_elmo/_{args.word}_{args.embedding_type}.json"
    )

db_path = "sqlite:////scratch/lh1036/embed_wiki_data/wikidb.db"
ft_path = f"/scratch/lh1036/embed_wiki_data/wiki.en.bin"
bert_path = f"/scratch/lh1036/uncased_L-12_H-768_A-12/"

word = args.word
db_name = "wikidb"
db_col = "text"

additional_params = {"whiten": True}
num_iterations = 50

dims = [2, 10, 50, 100, 200]
decomp_options = (
    list(zip(itertools.repeat("PCA", len(dims)), dims))
    + list(zip(itertools.repeat("UMAP", len(dims)), dims))
    + [("TSNE", 2)]
)

##########################################
if __name__ == "__main__":
    # pool = Pool(os.cpu_count() - 1)

    engine = create_engine(db_path)
    session = sessionmaker()
    session.configure(bind=engine)
    s = session()

    with open(output_fn, "w") as f:
        with contextlib.redirect_stdout(f):
            for context_option in [("sentence", None), ("window", 5), ("window", 10)]:
                for tf_idf_option in [False]:
                    for exclude_option in [False]:
                        for mask_pct in [None]:
                            context_type, window_size = context_option
                            print(f"# context_option {context_type} {window_size}\n")

                            contexts = Contexts(engine, word, db_name, db_col)

                            contexts.select_contexts(
                                context_type=context_type,
                                window_size=window_size,
                                num_contexts=1500,
                                randomized=False,
                            )

                            ce = ContextEmbeddings(
                                contexts,
                                tf_idf_weighting=tf_idf_option,
                                mask_pct=mask_pct,
                                exclude_word=exclude_option,
                            )

                            ce.extract_embeddings(
                                embedding_type=args.embedding_type,
                                bert_path=bert_path,
                                ft_path=ft_path,
                                subtract_context=True,
                            )
                            print(f"# {ce.embedding_representation.shape}")

                            for decomp_option in decomp_options:
                                decomp_method, decomp_dim = decomp_option

                                x = copy.deepcopy(ce)

                                if decomp_dim <= x.num_contexts:

                                    if decomp_method == "PCA":
                                        x.decompose_embeddings(
                                            decomp_method=decomp_method,
                                            decomp_dims=decomp_dim,
                                            additional_params={"whiten": True},
                                        )

                                    else:  # TSNE or UMAP
                                        x.decompose_embeddings(
                                            decomp_method=decomp_method,
                                            decomp_dims=decomp_dim,
                                        )

                                    bgms = [
                                        copy.deepcopy(x) for i in range(num_iterations)
                                    ]
                                    results = []

                                    for i in range(num_iterations):
                                        bgms[i].cluster_embeddings(
                                            cluster_method="BayesGaussMix",
                                            num_clusters=30,
                                            use_decomposed=True,
                                            additional_params={
                                                "random_state": i,
                                                "max_iter": 500,
                                            },
                                        )

                                        results.append(
                                            len(set(bgms[i].predicted_labels))
                                        )

                                    # results and all settings
                                    out = {
                                        "word": word,
                                        "context_type": x.context_type,
                                        "window_size": x.window_size,
                                        "num_contexts": x.num_contexts,
                                        "embedding_type": args.embedding_type,
                                        "decomp_method": decomp_method,
                                        "decomp_dims": decomp_dim,
                                        "tf_idf_weighting": tf_idf_option,
                                        "exclude_word": exclude_option,
                                        "mask_pct": mask_pct,
                                        "BGM_results": dict(Counter(results)),
                                    }

                                    print(f"{out},")

            print("done")
