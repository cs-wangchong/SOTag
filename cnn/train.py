# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import tensorflow as tf
import numpy as np
from model import TextCNN
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

tf.app.flags.DEFINE_integer("num_filters", 512, "number of filters")  # 256--->512
filter_sizes = [1, 2, 3, 4, 5, 6, 7]  # [1,2,3,4,5,6,7]
# 1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)


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
    print("num_filters:", FLAGS.num_filters)
    print("filter_sizes:", filter_sizes)

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
    # 2.Data preprocessing.Sequence padding
    print("Start padding & transform to one hot...")
    trainX = pad_sequences(trainX, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length
    validX = pad_sequences(validX, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length
    testX = pad_sequences(testX, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length
    print("End padding & transform to one hot...")
    # 2.create session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Instantiate Model
        textCNN = TextCNN(filter_sizes, FLAGS.num_filters, FLAGS.k, vocab_size, tag_vocab_size, FLAGS.learning_rate, FLAGS.decay_steps,
                          FLAGS.decay_rate, FLAGS.sentence_len, FLAGS.embed_size, FLAGS.is_training)
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
                assign_pretrained_word_embedding(sess, vocab, vocab_size, textCNN, word2vec_model_path=FLAGS.word2vec_model_path)
        curr_epoch = sess.run(textCNN.epoch_step)
        # 3.feed data & training
        number_of_training_data = len(trainX)
        batch_size = FLAGS.batch_size
        print("Training...")
        print("Start time: {}".format(time.asctime(time.localtime(time.time()))))
        for epoch in range(curr_epoch, FLAGS.num_epochs):
            print("Epoch: %d\tlr: %.3f" % (epoch, textCNN.lr))
            loss, recall, precision, counter = 0.0, 0.0, 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size), range(batch_size, number_of_training_data, batch_size)):
                # if epoch == 0 and counter == 0:
                #     print("trainX[start:end]:", trainX[start:end])  # ;print("trainY[start:end]:",trainY[start:end])
                feed_dict = {textCNN.input_x: trainX[start:end], textCNN.dropout_keep_prob: 0.5}
                input_y = [labels_2_onehot(y, tag_vocab_size) for y in trainY[start:end]]
                feed_dict[textCNN.input_y] = input_y
                curr_loss, topK, _ = sess.run([textCNN.loss_val, textCNN.topK, textCNN.train_op], feed_dict)  # curr_f1--->TextCNN.f1
                pred_nparr = np.zeros([batch_size, tag_vocab_size])
                for i in range(batch_size):
                    pred_nparr[i][topK[i]] = 1
                actual = [labels_2_onehot(y, tag_vocab_size) for y in trainY[start:end]]
                tp_op, fp_op, fn_op = sess.run([textCNN.tp_op, textCNN.fp_op, textCNN.fn_op], feed_dict={
                                               textCNN.pred: pred_nparr, textCNN.actual: actual})
                curr_recall = float(tp_op) / (float(tp_op) + float(fn_op))
                curr_precision = float(tp_op) / (float(tp_op) + float(fp_op))
                loss, recall, precision, counter = loss + curr_loss, recall + curr_recall, precision + curr_precision, counter + 1
                if counter % 50 == 0:
                    if recall == precision == 0:
                        f1 = 0.0
                    else:
                        temp_recall = recall / float(counter)
                        temp_precision = precision / float(counter)
                        f1 = (2 * (temp_precision * temp_recall)) / (temp_precision + temp_recall)
                    print("Batch %d\tTrain Loss:%.3f\tRecall:%.3f\tPrecision:%.3f\tF1:%.3f" %
                          (counter, loss / float(counter), temp_recall, temp_precision, f1))

            # 4.validation
            if epoch % FLAGS.validate_every == 0:
                eval_loss, eval_recall, eval_precision, eval_f1 = do_eval(sess, textCNN, validX, validY, batch_size, tag_vocab_size)
                print("Validation Loss:%.3f\tRecall:%.3f\tPrecision:%.3f\tF1:%.3f" %
                      (eval_loss, eval_recall, eval_precision, eval_f1))
                # save model to checkpoint
                save_path = path.join(ckpt_dir, "model.ckpt")
                saver.save(sess, save_path, global_step=epoch)

            _, lr = sess.run([textCNN.epoch_increment, textCNN.lr_decay])
            textCNN.lr = lr
        print("End time {}".format(time.asctime(time.localtime(time.time()))))
        # 5.Test
        test_loss, test_recall, test_precision, test_f1 = do_eval(sess, textCNN, testX, testY, batch_size, tag_vocab_size)
        print("Test Loss:%.3f\tRecall: %.3f\tPrecision: %.3f\tF1: %.3f" % (test_loss, test_recall, test_precision, test_f1))
    pass


def assign_pretrained_word_embedding(sess, vocab, vocab_size, textCNN, word2vec_model_path=None):
    print("Using pre-trained word emebedding. Path:", word2vec_model_path)
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
    t_assign_embedding = tf.assign(textCNN.Embedding, word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding)
    print("Word exists embedding:", count_exist, "; Word not exist embedding:", count_not_exist)

# 在验证集上做验证，报告损失、精确度


def do_eval(sess, textCNN, evalX, evalY, batch_size, tag_vocab_size):
    number_examples = len(evalX)
    eval_loss, recall, precision, eval_counter = 0.0, 0.0, 0.0, 0
    for start, end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples, batch_size)):
        feed_dict = {textCNN.input_x: evalX[start:end], textCNN.dropout_keep_prob: 1}
        input_y = [labels_2_onehot(y, tag_vocab_size) for y in evalY[start:end]]
        feed_dict[textCNN.input_y] = input_y
        curr_eval_loss, topK = sess.run([textCNN.loss_val, textCNN.topK], feed_dict)  # curr_f1--->TextCNN.f1
        pred_nparr = np.zeros([batch_size, tag_vocab_size])
        for i in range(batch_size):
            pred_nparr[i][topK[i]] = 1
        actual = [labels_2_onehot(y, tag_vocab_size) for y in evalY[start:end]]
        tp_op, fp_op, fn_op = sess.run([textCNN.tp_op, textCNN.fp_op, textCNN.fn_op], feed_dict={
            textCNN.pred: pred_nparr, textCNN.actual: actual})
        curr_recall = float(tp_op) / (float(tp_op) + float(fn_op))
        curr_precision = float(tp_op) / (float(tp_op) + float(fp_op))
        # label_list_top5 = get_label_using_logits(logits_[0], vocabulary_index2word_label)
        # curr_eval_f1=calculate_accuracy(list(label_list_top5), evalY[start:end][0],eval_counter)
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
