import sys
sys.path.append("..")
sys.path.append(".")
from tagRec.data3 import load_tag_vocab, load_vocab
import tensorflow as tf
from fastText.model import FastText
from tflearn.data_utils import pad_sequences
from os import path
import re
from nltk.stem import WordNetLemmatizer
import nltk
import csv
import numpy as np


sess = None
fastText = None
k = 5
sentence_len = 100
embed_size = 128
ckpt_dir = path.join(path.dirname(__file__), 'ckpt')


def init(vocab_size, tag_vocab_size):

    # 2.create session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    global sess
    sess = tf.Session(config=config)
    with sess.as_default():
        # Instantiate Model
        global fastText
        fastText = FastText(k, vocab_size, tag_vocab_size, 0.0, 0.0,
                            0.0, embed_size, sentence_len, False)
        # Initialize Save
        saver = tf.train.Saver()
        if path.exists(path.join(ckpt_dir, "checkpoint")):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
        else:
            print("Can't find the checkpoint. Going to stop")
            exit(1)


def preprocess(title, question, vocab, vocab_index, stoplist):
    x = []
    wnl = WordNetLemmatizer()
    # sentences = []
    pattern1 = re.compile(r'<code>.*?</code>')
    pattern2 = re.compile(r'<code>.*?</code>|<blockquote>.*?</blockquote>|<a.*?</a>|</?.*?>|&#x[AD];|\\t|\\n')
    pattern3 = re.compile(r'</?code>|&#x[AD];|\\t|\\n')
    pattern4 = re.compile(u'[0-9]*')
    for word in re.split('[ ,]', title):
        w = wnl.lemmatize(word.lower().strip(' ’!"$%&\'()*,-./:;<=>?‘“”？，；…@[\\]^_`{|}~'))
        if pattern4.fullmatch(w) is None and w not in stoplist:
            x.append(w)

    code_list = pattern1.findall(question)
    for i in range(len(code_list)):
        code_list[i] = pattern3.sub("", code_list[i])
    body = pattern2.sub(" ", question)
    sents = nltk.sent_tokenize(body)
    for sent in sents:
        words = re.split('[ ,]', sent)
        new_words = []
        for word in words:
            w = wnl.lemmatize(word.lower().strip(' ’!"$%&\'()*,-./:;<=>?‘“”？，；…@[\\]^_`{|}~'))
            if pattern4.fullmatch(w) is None and w not in stoplist:
                new_words.append(w)
        if len(new_words) > 0:
            x.extend(new_words)
    x_ = []
    for word in x:
        if word in vocab:
            x_.append(vocab_index[word])
        else:
            x_.append(0)
    return x_


def predict(words, tag_vocab):
    if len(words) == 0:
        return tag_vocab[:5]
    input_x = pad_sequences([words], maxlen=sentence_len, value=0.)
    input_length = np.array([[len(words)]])
    feed_dict = {fastText.input_x: input_x, fastText.input_length: input_length, fastText.dropout_keep_prob: 1}
    with sess.as_default():
        topK = sess.run([fastText.topK], feed_dict)
    tags = [tag_vocab[index] for index in topK[0][0]]

    return tags


if __name__ == "__main__":
    vocab, vocab_index = load_vocab()
    tag_vocab, tag_index = load_tag_vocab()
    vocab_size = len(vocab)
    tag_vocab_size = len(tag_vocab)
    init(vocab_size, tag_vocab_size)
    processed_dir = path.join(path.dirname(__file__), "../dataset/processed2")
    stoplist_csv = path.join(processed_dir, "stoplist.csv")
    with open(stoplist_csv, "r", encoding='UTF-8') as csv_file:
        reader = csv.reader(csv_file)
        reader.__next__()
        stoplist = [row[0] for row in reader]
    while True:
        title = input("title:\n")
        if title == "quit":
            if sess is not None:
                sess.close()
            break
        question = input("question:\n")
        words = preprocess(title, question, vocab, vocab_index, stoplist)
        tags = predict(words, tag_vocab)
        print("tags:")
        print(tags)
        print("\n")
