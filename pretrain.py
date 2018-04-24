#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from os import path
import csv
import json
import time


processed_dir = path.join(path.dirname(__file__), "./dataset/processed3")
corpora_csv = path.join(processed_dir, "corpora-core10.csv")
vocab_json = path.join(processed_dir, "vocab-core10.json")

sentences = []
with open(corpora_csv, "r", encoding='UTF-8') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        sentences.append(row)
with open(vocab_json, "r", encoding='UTF-8') as json_file:
    vocab_dict = json.load(json_file)
print("Training...")
print("start time {}".format(time.asctime(time.localtime(time.time()))))
model = Word2Vec(sentences, size=128, window=10, min_count=9, sg=1, hs=1, workers=4)
print("end time {}".format(time.asctime(time.localtime(time.time()))))
model.save("./model/Word2Vec-1m.model")
model.wv.save("./model/vocab-1m.txt")
# model = Word2Vec.load(".\\model\\Word2Vec.model")
# print(model.vocab)
# model.wv.save(".\\model\\vocab.txt")
# wv = KeyedVectors.load(".\\model\\vocab.txt")

# try:
#     print(wv["n't"])
# except KeyError:
#     print("no word")
