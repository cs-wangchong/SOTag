# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import tensorflow as tf
import numpy as np
from model import TextRNN
from gensim.models import KeyedVectors
from tflearn.data_utils import pad_sequences
from os import path
from tagRec.data2 import load_tag_vocab, load_vocab, load_dataset
import time

# configuration
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float("learning_rate", 0.05, "learning rate")
tf.app.flags.DEFINE_integer("batch_size", 512, "Batch size for training/evaluating.")  # 批处理的大小 32-->128
tf.app.flags.DEFINE_integer("k", 5, "Top k.")  # 批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 20, "how many steps before decay learning rate.")  # 6000批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 0.1, "Rate of decay for learning rate.")  # 0.65一次衰减多少
# tf.app.flags.DEFINE_integer("num_sampled",50,"number of noise sampling") #100
tf.app.flags.DEFINE_string("ckpt_dir", "ckpt", "checkpoint location for the model")
tf.app.flags.DEFINE_string("word2vec_model_path", "./model/vocab.txt", "checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_len", 100, "max sentence length")
tf.app.flags.DEFINE_integer("embed_size", 128, "embedding size")
tf.app.flags.DEFINE_boolean("is_training", True, "is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs", 20, "number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")  # 每10轮做一次验证
tf.app.flags.DEFINE_boolean("use_embedding", True, "whether to use embedding or not.")

def main(_):
    print("learning_rate:", FLAGS.learning_rate)
    print("batch_size:", FLAGS.batch_size)
    print("k:", FLAGS.k)
    print("decay_steps:", FLAGS.decay_steps)
    print("decay_rate:", FLAGS.decay_rate)
    print("ckpt_dir:", FLAGS.ckpt_dir)
    print("word2vec_model_path:", FLAGS.word2vec_model_path)
    print("sentence_len:", FLAGS.sentence_len)
    print("embed_size:", FLAGS.embed_size)
    print("num_epochs:", FLAGS.num_epochs)
    print("validate_every:", FLAGS.validate_every)
    print("use_embedding:", FLAGS.use_embedding)

    trainX, trainY, validX, validY, testX, testY = None, None, None, None, None, None
    vocab, vocab_index = load_vocab()
    vocab_size = len(vocab)
    print("vocab_size:", vocab_size)

    tag_vocab, tag_index = load_tag_vocab()
    tag_vocab_size = len(tag_vocab)
    print("tag_vocab_size:", tag_vocab_size)
    train, valid, test = load_dataset(vocab_index, tag_index, tag_vocab_size)  # ,traning_data_path=FLAGS.traning_data_path
    trainX, trainY = train
    validX, validY = valid
    testX, testY = test
    print("trainset_size:", len(trainX))
    train_sent_len = [len(x) if len(x) <= 100 else 100 for x in trainX]
    valid_sent_len = [len(x) if len(x) <= 100 else 100 for x in validX]
    test_sent_len = [len(x) if len(x) <= 100 else 100 for x in testX]
    print("Start padding & transform to one hot...")
    trainX = pad_sequences(trainX, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length
    validX = pad_sequences(validX, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length
    testX = pad_sequences(testX, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length
    print("End padding & transform to one hot...")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Instantiate Model
        textRNN = TextRNN(FLAGS.k, vocab_size, tag_vocab_size, FLAGS.learning_rate, FLAGS.decay_steps,
                          FLAGS.decay_rate, FLAGS.embed_size, FLAGS.sentence_len, FLAGS.is_training)
        # Initialize Save
        saver = tf.train.Saver()
        ckpt_dir = path.join(path.dirname(__file__), FLAGS.ckpt_dir)
        if path.exists(path.join(ckpt_dir, "checkpoint")):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding:  # load pre-trained word embedding
                load_pretrained_embed(sess, vocab, vocab_size, textRNN, word2vec_model_path=FLAGS.word2vec_model_path)
        curr_epoch = sess.run(textRNN.epoch_step)
        # 3.feed data & training
        number_of_training_data = len(trainX)
        batch_size = FLAGS.batch_size
        print("Training...")
        print("Start time {}".format(time.asctime(time.localtime(time.time()))))
        for epoch in range(curr_epoch, FLAGS.num_epochs):
            print("Epoch: %d\tlr: %.3f" % (epoch, textRNN.lr))
            loss, recall, precision, counter = 0.0, 0.0, 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size), range(batch_size, number_of_training_data, batch_size)):

                feed_dict = {textRNN.input_x: trainX[start:end], textRNN.input_length: train_sent_len[start:end], textRNN.dropout_keep_prob: 1.0}
                input_y = [labels_2_onehot(y, tag_vocab_size) for y in trainY[start:end]]
                feed_dict[textRNN.input_y] = input_y
                curr_loss, topK, _ = sess.run([textRNN.loss_val, textRNN.topK, textRNN.train_op], feed_dict)  # curr_f1--->textRNN.f1
                pred_nparr = np.zeros([batch_size, tag_vocab_size])
                for i in range(batch_size):
                    pred_nparr[i][topK[i]] = 1
                actual = [labels_2_onehot(y, tag_vocab_size) for y in trainY[start:end]]
                tp_op, fp_op, fn_op = sess.run([textRNN.tp_op, textRNN.fp_op, textRNN.fn_op], feed_dict={
                                               textRNN.pred: pred_nparr, textRNN.actual: actual})
                curr_recall = float(tp_op) / (float(tp_op) + float(fn_op))
                curr_precision = float(tp_op) / (float(tp_op) + float(fp_op))

                loss, recall, precision, counter = loss + curr_loss, recall + curr_recall, precision + curr_precision, counter + 1
                if counter % 50 == 0:
                # if counter > 100 and counter < 150:
                    if recall == precision == 0:
                        f1 = 0.0
                    else:
                        temp_recall = recall / float(counter)
                        temp_precision = precision / float(counter)
                        f1 = (2 * (temp_precision * temp_recall)) / (temp_precision + temp_recall)
                    print("Batch %d\tTrain Loss:%.3f\tRecall:%.3f\tPrecision:%.3f\tF1:%.3f" %
                          (counter, loss / float(counter), temp_recall, temp_precision, f1))

            # 4.Validation
            if epoch % FLAGS.validate_every == 0:
                eval_loss, eval_recall, eval_precision, eval_f1 = do_eval(sess, textRNN, validX, validY, valid_sent_len, batch_size, tag_vocab_size)
                print("Validation Loss:%.3f\tRecall:%.3f\tPrecision:%.3f\tF1:%.3f" %
                      (eval_loss, eval_recall, eval_precision, eval_f1))
                # save model to checkpoint
                save_path = path.join(ckpt_dir, "model.ckpt")
                saver.save(sess, save_path, global_step=epoch)
            _, lr = sess.run([textRNN.epoch_increment, textRNN.lr_decay])
            textRNN.lr = lr
        print("End time {}".format(time.asctime(time.localtime(time.time()))))
        # 5.Test
        test_loss, test_recall, test_precision, test_f1 = do_eval(sess, textRNN, testX, testY, test_sent_len, batch_size, tag_vocab_size)
        print("Test Loss:%.3f\tRecall: %.3f\tPrecision: %.3f\tF1: %.3f" % (test_loss, test_recall, test_precision, test_f1))
    pass


def load_pretrained_embed(sess, vocab, vocab_size, textRNN, word2vec_model_path=None):
    print("Using pre-trained word emebedding. Path: ", word2vec_model_path)
    # word2vecc=word2vec.load('word_embedding.txt') #load vocab-vector fiel.word2vecc['w91874']
    wv = KeyedVectors.load(word2vec_model_path)

    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0
    count_not_exist = 0
    for i in range(1, vocab_size):  # loop each word
        word = vocab[i]  # get a word
        embedding = None
        try:
            embedding = wv[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size)
            count_not_exist = count_not_exist + 1  # init a random value for the word.
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(textRNN.Embedding, word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding)
    print("Word exists embedding:", count_exist, "; Word not exist embedding:", count_not_exist)


def do_eval(sess, textRNN, evalX, evalY, sent_len, batch_size, tag_vocab_size):
    number_examples = len(evalX)
    eval_loss, recall, precision, eval_counter = 0.0, 0.0, 0.0, 0
    for start, end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples, batch_size)):
        feed_dict = {textRNN.input_x: evalX[start:end], textRNN.input_length: sent_len[start:end], textRNN.dropout_keep_prob: 1}
        input_y = [labels_2_onehot(y, tag_vocab_size) for y in evalY[start:end]]
        feed_dict[textRNN.input_y] = input_y
        curr_eval_loss, topK = sess.run([textRNN.loss_val, textRNN.topK], feed_dict)  # curr_f1--->textRNN.f1
        pred_nparr = np.zeros([batch_size, tag_vocab_size])
        for i in range(batch_size):
            pred_nparr[i][topK[i]] = 1
        actual = [labels_2_onehot(y, tag_vocab_size) for y in evalY[start:end]]
        tp_op, fp_op, fn_op = sess.run([textRNN.tp_op, textRNN.fp_op, textRNN.fn_op], feed_dict={
            textRNN.pred: pred_nparr, textRNN.actual: actual})
        curr_recall = float(tp_op) / (float(tp_op) + float(fn_op))
        curr_precision = float(tp_op) / (float(tp_op) + float(fp_op))
        eval_loss, recall, precision, eval_counter = eval_loss + curr_eval_loss, recall + curr_recall, precision + curr_precision, eval_counter + 1
    if recall == precision == 0:
        f1 = 0.0
    else:
        temp_recall = recall / float(eval_counter)
        temp_precision = precision / float(eval_counter)
        f1 = (2 * (temp_precision * temp_recall)) / (temp_precision + temp_recall)
    return eval_loss / float(eval_counter), temp_recall, temp_precision, f1

def labels_2_probs(label_list, label_size=100):  # 1999label_list=[0,1,4,9,5]
    num = len(label_list)
    result = np.zeros(label_size)
    # set those location as 1, all else place as 0.
    result[label_list] = 1. / num
    return result


def labels_2_onehot(label_list, label_size=100):  # 1999label_list=[0,1,4,9,5]
    result = np.zeros(label_size)
    # set those location as 1, all else place as 0.
    result[label_list] = 1.
    return result


if __name__ == "__main__":
    tf.app.run()
