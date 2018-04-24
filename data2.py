#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pymysql as mdb
import csv
from os import path
import re
from nltk.stem import WordNetLemmatizer
import nltk
import json
import random


original_dir = path.join(path.dirname(__file__), "./dataset/original2")
processed_dir = path.join(path.dirname(__file__), "./dataset/processed2")
posts_csv = path.join(original_dir, "posts.csv")
corpora_csv = path.join(processed_dir, "corpora.csv")
corpora_csv_core3 = path.join(processed_dir, "corpora-core3.csv")
posts_json = path.join(processed_dir, "posts.json")
posts_json_core5 = path.join(processed_dir, "posts-core5.json")

stoplist_csv = path.join(processed_dir, "stoplist.csv")

vocab_json = path.join(processed_dir, "vocab.json")
vocab_json_core3 = path.join(processed_dir, "vocab-core3.json")
tags_json = path.join(processed_dir, "tags.json")
tags_json_core20 = path.join(processed_dir, "tags-core20.json")

testdataX_json = path.join(processed_dir, "testdataX.json")
testdataY_json = path.join(processed_dir, "testdataY.json")


def download():
    conn = mdb.connect(host='10.131.252.160', port=3306, user='root',
                       passwd='root', db='stackoverflow', charset='utf8')
    cursor = conn.cursor()
    cursor.execute(
        "SELECT Id, Body, Title, Tags FROM posts \
        WHERE score >= 20 AND Tags IS NOT NULL")
    r = cursor.fetchall()

    headers = ["Id", "Body", "Title", "Tags"]
    with open(posts_csv, "w", newline="", encoding='UTF-8') as csv_file:
        writer = csv.writer(csv_file, dialect='excel')
        writer.writerow(headers)
        writer.writerows(r)


def preprocess():
    vocab = {}
    tag_vocab = {}
    wnl = WordNetLemmatizer()
    processed_data = []
    # sentences = []
    pattern1 = re.compile(r'<code>.*?</code>')
    pattern2 = re.compile(r'<code>.*?</code>|<blockquote>.*?</blockquote>|<a.*?</a>|</?.*?>|&#x[AD];|\\t|\\n')
    pattern3 = re.compile(r'</?code>|&#x[AD];|\\t|\\n')
    pattern4 = re.compile(u'[0-9]*')
    with open(stoplist_csv, "r", encoding='UTF-8') as csv_file:
        reader = csv.reader(csv_file)
        reader.__next__()
        stoplist = [row[0] for row in reader]
    with open(posts_csv, "r", encoding='UTF-8') as csv_file:
        reader = csv.reader(csv_file)
        reader.__next__()
        for row in reader:
            code_list = pattern1.findall(row[1])
            for i in range(len(code_list)):
                code_list[i] = pattern3.sub("", code_list[i])
            body = pattern2.sub(" ", row[1])
            sents = nltk.sent_tokenize(body)
            new_body = []
            for sent in sents:
                words = re.split('[ ,]', sent)
                new_words = []
                for word in words:
                    w = wnl.lemmatize(word.lower().strip(' ’!"$%&\'()*,-./:;<=>?‘“”？，；…@[\\]^_`{|}~'))
                    if pattern4.fullmatch(w) is None and w not in stoplist:
                        if vocab.get(w, None) is not None:
                            vocab[w] += 1
                        else:
                            vocab[w] = 1
                        new_words.append(w)
                if len(new_words) > 0:
                    # sentences.append(new_words)
                    new_body.extend(new_words)
            new_title = []
            for word in re.split('[ ,]', row[2]):
                w = wnl.lemmatize(word.lower().strip(' ’!"$%&\'()*,-./:;<=>?‘“”？，；…@[\\]^_`{|}~'))
                if pattern4.fullmatch(w) is None and w not in stoplist:
                    if vocab.get(w, None) is not None:
                        vocab[w] += 1
                    else:
                        vocab[w] = 1
                    new_title.append(w)
            # if len(new_title) > 0:
            #     sentences.append(new_title)
            new_tags = [wnl.lemmatize(e.strip('<> ').lower()) for e in row[3].split('><')]
            processed_data.append([row[0], new_body, new_title, new_tags, code_list])
            for tag in new_tags:
                if tag_vocab.get(tag, None) is not None:
                    tag_vocab[tag] += 1
                else:
                    tag_vocab[tag] = 1
    with open(vocab_json, "w", newline="", encoding='UTF-8') as json_file:
        json.dump(sort_by_value(vocab), json_file)
    with open(tags_json, "w", newline="", encoding='UTF-8') as json_file:
        json.dump(sort_by_value(tag_vocab), json_file)
    # with open(corpora_csv, "w", newline="", encoding='UTF-8') as csv_file:
    #     writer = csv.writer(csv_file, dialect='excel')
    #     writer.writerows(sentences)
    random.shuffle(processed_data)
    with open(posts_json, "w", newline="", encoding='UTF-8') as json_file:
        json.dump(processed_data, json_file)
    # headers = ["Id", "Body", "OwnerUserId", "Title", "Tags", "CreationDate", "codes"]
    # with open(posts_csv_p, "w", newline="", encoding='UTF-8') as csv_file:
    #     writer = csv.writer(csv_file, dialect='excel')
    #     writer.writerow(headers)
    #     writer.writerows(processed_data)


def filterData():
    vocab_dcount = {}
    with open(posts_json, "r", encoding='UTF-8') as json_file:
        dataset_raw = json.load(json_file)
    for data in dataset_raw:
        d_words = []
        for word in data[1]:
            if vocab_dcount.get(word, None) is None:
                vocab_dcount[word] = 1
                d_words.append(word)
            else:
                if word not in d_words:
                    vocab_dcount[word] += 1
                    d_words.append(word)
        for word in data[2]:
            if vocab_dcount.get(word, None) is None:
                vocab_dcount[word] = 1
                d_words.append(word)
            else:
                if word not in d_words:
                    vocab_dcount[word] += 1
                    d_words.append(word)
    with open(vocab_json, "r", encoding='UTF-8') as json_file:
        vocab_dict = json.load(json_file)
    vocab = {}
    pattern1 = re.compile(r'[a-zA-Z]*[0-9+-:#]+')
    pattern2 = re.compile(r'[/\\\*\(\),=\?<>"\[\]]|0x')
    pattern3 = re.compile(r'([a-zA-Z])\1{3,}')
    for word, count in vocab_dict.items():
        # if count >= 3 and pattern1.match(word) is None and pattern2.search(word) is None:
        #     vocab[word] = count
        if count < 3 or (pattern1.match(word) is not None or pattern2.search(word) is not None or pattern3.search(word) is not None) or (count <= 5 and vocab_dcount[word] == 1):
            continue
        vocab[word] = count
    with open(tags_json, "r", encoding='UTF-8') as json_file:
        tag_vocab_dict = json.load(json_file)
    tag_vocab = {}
    for tag, count in tag_vocab_dict.items():
        if count >= 20:
            tag_vocab[tag] = count

    dataset = []
    for data in dataset_raw:
        body = []
        title = []
        tags = []
        for tag in data[3]:
            if tag_vocab.get(tag, None) is not None:
                tags.append(tag)
        if len(tags) == 0:
            continue
        for word in data[1]:
            if vocab.get(word, None) is not None:
                body.append(word)
        for word in data[2]:
            if vocab.get(word, None) is not None:
                title.append(word)
        if len(body) + len(title) > 0:
            dataset.append([data[0], body, title, tags, data[4]])
    # with open(corpora_csv, "r", encoding='UTF-8') as csv_file:
    #     reader = csv.reader(csv_file)
    #     sentences_raw = [row for row in reader]
    # sentences = []
    # for sent in sentences_raw:
    #     new_sent = []
    #     for word in sent:
    #         if vocab.get(word, None) is not None:
    #             new_sent.append(word)
    #     sentences.append(new_sent)
    with open(posts_json_core5, "w", newline="", encoding='UTF-8') as json_file:
        json.dump(dataset, json_file)
    with open(tags_json_core20, "w", newline="", encoding='UTF-8') as json_file:
        json.dump(tag_vocab, json_file)
    with open(vocab_json_core3, "w", newline="", encoding='UTF-8') as json_file:
        json.dump(vocab, json_file)   
    # with open(corpora_csv_core3, "w", newline="", encoding='UTF-8') as csv_file:
    #     writer = csv.writer(csv_file, dialect='excel')
    #     writer.writerows(sentences)


def gen_corpora(valid_portion=0.1):
    sentences = []

    with open(posts_json_core5, "r", encoding='UTF-8') as json_file:
        dataset = json.load(json_file)
    dataset_size = len(dataset)
    for data in dataset[:int((1 - 2 * valid_portion) * dataset_size)]:
        sentences.append(data[1])
        sentences.append(data[2])
    with open(corpora_csv_core3, "w", newline="", encoding='UTF-8') as csv_file:
        writer = csv.writer(csv_file, dialect='excel')
        writer.writerows(sentences)


def load_vocab():
    vocab = []
    vocab_index = {}
    with open(vocab_json_core3, "r", encoding='UTF-8') as json_file:
        vocab_dict = json.load(json_file)
        vocab = list(vocab_dict.keys())
        for i, word in enumerate(vocab):
            vocab_index[word] = i
    vocab.insert(0, "PAD")
    vocab_index["PAD"] = 0
    return vocab, vocab_index


def load_tag_vocab():
    tag_vocab = []
    tag_index = {}
    with open(tags_json_core20, "r", encoding='UTF-8') as json_file:
        tag_vocab_dict = json.load(json_file)
        tag_vocab = list(tag_vocab_dict.keys())
        for i, tag in enumerate(tag_vocab):
            tag_index[tag] = i
    return tag_vocab, tag_index


def load_dataset(vocab_index, tag_index, tag_vocab_size, valid_portion=0.1):
    X, Y = [], []
    # with open(posts_csv_p, "r", encoding='UTF-8') as csv_file:
    #     reader = csv.reader(csv_file)
    #     reader.__next__()
    #     dataset = [row for row in reader]
    with open(posts_json_core5, "r", encoding='UTF-8') as json_file:
        dataset = json.load(json_file)
    tag_count = 0
    for i, data in enumerate(dataset):
        x, y = [], []
        for word in data[2]:
            x.append(vocab_index[word])
        for word in data[1]:
            x.append(vocab_index[word])
        for tag in data[3]:
            y.append(tag_index[tag])
        # ys_mulithot_list = transform_multilabel_as_multihot(y, tag_vocab_size)
        tag_count += len(y)
        X.append(x)
        Y.append(y)

    dataset_size = len(X)
    print("dataset size: %i" % dataset_size)
    print("average tag count: %3f" % (tag_count / dataset_size))
    train = (X[0:int((1 - 2 * valid_portion) * dataset_size)], Y[0:int((1 - 2 * valid_portion) * dataset_size)])
    valid = (X[int((1 - 2 * valid_portion) * dataset_size) + 1:int((1 - valid_portion) * dataset_size)],
             Y[int((1 - 2 * valid_portion) * dataset_size) + 1:int((1 - valid_portion) * dataset_size)])
    test = (X[int((1 - valid_portion) * dataset_size) + 1:], Y[int((1 - valid_portion) * dataset_size) + 1:])
    with open(testdataX_json, "w", newline="", encoding='UTF-8') as json_file:
        json.dump(test[0], json_file)
    with open(testdataY_json, "w", newline="", encoding='UTF-8') as json_file:
        json.dump(test[1], json_file)
    return train, valid, test


def sort_by_value(d):
    items = d.items()
    backitems = [[v[1], v[0]] for v in items]
    backitems.sort(reverse=True)
    new_d = {}
    for item in backitems:
        new_d[item[1]] = item[0]
    return new_d


if __name__ == "__main__":
    # download()
    # preprocess()
    filterData()
    gen_corpora()
    # vocab, vocab_index = load_vocab()
    # vocab_size = len(vocab)
    # print("vocab_size:", vocab_size)

    # tag_vocab, tag_index = load_tag_vocab()
    # tag_vocab_size = len(tag_vocab)
    # print("tag_vocab_size:", tag_vocab_size)
    # load_dataset(vocab_index, tag_index, tag_vocab_size)
