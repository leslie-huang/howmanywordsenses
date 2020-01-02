#!/usr/bin/env python3

from collections import OrderedDict
from contexts import *
import contextlib
from PyDictionary import PyDictionary
from wiktionaryparser import WiktionaryParser

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

db_path = "sqlite:////scratch/lh1036/embed_wiki_data/wikidb.db"
# db_path = "sqlite:///../embed_wiki_data/wikidb.db"
# db_path = "sqlite:///embed_parliament_data/hansard1970sdb.db"
engine = create_engine(db_path)

session = sessionmaker()
session.configure(bind=engine)
s = session()

pydict = PyDictionary()
parser = WiktionaryParser()

output_fn = "/home/lh1036/topwords_wiki.json"
# output_fn = "/home/lh1036/topwords_hansard.json"

# all_contexts = Contexts(engine, "", "hansard1970s", "text")
all_contexts = Contexts(engine, "", "wikidb", "text")

top_2000 = get_corpus_top_words(all_contexts, 2000)  # list of (word, count) tuples

top_words = OrderedDict()

for pair in top_2000:
    word, count = pair

    pyd_count = None
    wiki_count = None

    if word.isalpha():
        print(f"=============\n{word}")
        try:
            pyd_count = sum([len(x[1]) for x in pydict.meaning(word).items()])
        except:
            print(f"PyDict can't find {word}")

        if pyd_count:
            try:
                wiktionary_word = parser.fetch(word)[0]
                if len(wiktionary_word["definitions"]) != 0:
                    wiki_count = sum(
                        [len(i["text"]) for i in wiktionary_word["definitions"]]
                    )
                else:
                    print(f"Zero Wiktionary definitions for {word}")
            except:
                print(f"Wiktionary failed on {word}")

            if wiki_count:
                top_words[word] = {
                    "pydict": pyd_count,
                    "wiktionary": wiki_count,
                    "frequency": count,
                }

with open(output_fn, "w") as f:
    json.dump(top_words, f)
