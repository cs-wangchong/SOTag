from flask import Flask, request, render_template
from os import path
import csv
import json
import re
from nltk.stem import WordNetLemmatizer
import nltk
from data2 import load_tag_vocab, load_vocab
import cnn.predict as cnn_predict
import fastText.predict as fastText_predict

app = Flask(__name__, static_folder='./tmp')

vocab, vocab_index = load_vocab()
tag_vocab, tag_index = load_tag_vocab()

processed_dir = path.join(path.dirname(__file__), "./dataset/processed2")
stoplist_csv = path.join(processed_dir, "stoplist.csv")
with open(stoplist_csv, "r", encoding='UTF-8') as csv_file:
    reader = csv.reader(csv_file)
    reader.__next__()
    stoplist = [row[0] for row in reader]


def preprocess(title, question):
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


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/cnn/tags')
def get_tags_cnn():
    title = request.args.get('title')
    question = request.args.get('question')
    words = preprocess(title, question)
    tags = cnn_predict.predict(words, tag_vocab)
    return json.dumps(tags)


@app.route('/fastText/tags')
def get_tags_fast():
    title = request.args.get('title')
    question = request.args.get('question')
    words = preprocess(title, question)
    tags = fastText_predict.predict(words, tag_vocab)
    return json.dumps(tags)


if __name__ == "__main__":
    vocab_size = len(vocab)
    print("vocab_size:", vocab_size)
    tag_vocab_size = len(tag_vocab)
    print("tag_vocab_size:", tag_vocab_size)

    cnn_predict.init(vocab_size, tag_vocab_size)
    # fastText_predict.init(vocab_size, tag_vocab_size)
    app.run()
