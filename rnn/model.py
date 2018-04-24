# -*- coding: utf-8 -*-
# TextRNN: 1. embeddding layer, 2.Bi-LSTM layer, 3.concat output, 4.FC layer, 5.softmax
import tensorflow as tf
from tensorflow.contrib import rnn


class TextRNN:
    def __init__(self, k, vocab_size, num_classes, learning_rate, decay_steps, decay_rate, embed_size, sequence_length,
                 is_training, initializer=tf.random_normal_initializer(stddev=0.1), clip_gradients=5.0, decay_rate_big=0.50):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = embed_size
        self.is_training = is_training
        self.initializer = initializer
        self.clip_gradients = clip_gradients

        # add placeholder (X,label)
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  # X
        self.input_length = tf.placeholder(tf.int32, [None], name="input_length")  # X
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")  # y [None,num_classes]
        self.actual = tf.placeholder(tf.float32, [None, self.num_classes], name="actual")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.pred = tf.placeholder(tf.int32, [None, self.num_classes], name="pred")  # X

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.logits = self.inference()  # [None, self.label_size]. main computation graph is here.
        self.topK = tf.nn.top_k(self.logits, k).indices
        self.tp_op, self.fp_op, self.fn_op = self.calc_f1()
        if not is_training:
            return
        self.lr = learning_rate
        self.lr_decay = tf.train.exponential_decay(learning_rate, self.epoch_step, self.decay_steps, self.decay_rate, staircase=False)
        self.loss_val = self.loss()  # -->self.loss_nce()
        self.train_op = self.train()

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding"):  # embedding matrix
            # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size], initializer=self.initializer)
            self.W_projection = tf.get_variable("W_projection", shape=[self.hidden_size * 2,
                                                                       self.num_classes], initializer=self.initializer)  # [embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])  # [label_size]

    def inference(self):
        """main computation graph here: 1. embeddding layer, 2.Bi-LSTM layer, 3.concat, 4.FC layer 5.softmax """
        # 1.get emebedding of words in the sentence
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)  # shape:[None,sentence_length,embed_size]
        # 2. Bi-lstm layer
        # define lstm cess:get lstm cell output
        lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)  # forward direction cell
        lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)  # backward direction cell
        if self.dropout_keep_prob is not None:
            lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob)
            lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob)
        # bidirectional_dynamic_rnn: input: [batch_size, max_time, input_size]
        #                            output: A tuple (outputs, output_states)
        #                                    where:outputs: A tuple (output_fw, output_bw) containing the forward and the backward rnn output `Tensor`.
        # [batch_size,sequence_length,hidden_size] #creates a dynamic bidirectional recurrent neural network
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.embedded_words, sequence_length=self.input_length, dtype=tf.float32)
        # outputs:(<tf.Tensor 'bidirectional_rnn/fw/fw/transpose:0' shape=(?, 5, 100) dtype=float32>, <tf.Tensor 'ReverseV2:0' shape=(?, 5, 100) dtype=float32>))
        print("outputs:===>", outputs)
        # 3. concat output
        output_rnn = tf.concat(outputs, axis=2)  # [batch_size,sequence_length,hidden_size*2]
        # [batch_size,hidden_size*2] #output_rnn_last=output_rnn[:,-1,:] ##[batch_size,hidden_size*2] #TODO
        self.output_rnn_last = tf.reduce_mean(output_rnn, axis=1)
        print("output_rnn_last:", self.output_rnn_last)  # <tf.Tensor 'strided_slice:0' shape=(?, 200) dtype=float32>
        # 4. logits(use linear layer)
        with tf.name_scope("output"):  # inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
            logits = tf.matmul(self.output_rnn_last, self.W_projection) + self.b_projection  # [batch_size,num_classes]
        return logits

    def loss(self, l2_lambda=0.00001):
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            losses = tf.reduce_sum(losses, axis=1)  # shape=(?,). loss for all data in the batch
            print("Use sigmoid_cross_entropy_with_logits.")
            # losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            # print("Use softmax_cross_entropy_with_logits.")
            loss = tf.reduce_mean(losses)  # shape=().   average loss in the batch
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, learning_rate=self.lr, global_step=self.global_step, optimizer="Adam", clip_gradients=self.clip_gradients)
        return train_op

    def calc_f1(self):
        ones_like_actuals = tf.ones_like(self.actual)
        zeros_like_actuals = tf.zeros_like(self.actual)
        ones_like_predictions = tf.ones_like(self.pred)
        zeros_like_predictions = tf.zeros_like(self.pred)

        tp_op = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(self.actual, ones_like_actuals),
                    tf.equal(self.pred, ones_like_predictions)
                ),
                "float"
            )
        )
        fp_op = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(self.actual, zeros_like_actuals), 
                    tf.equal(self.pred, ones_like_predictions)
                ), 
            "float"
            )
        )
        fn_op = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(self.actual, ones_like_actuals), 
                    tf.equal(self.pred, zeros_like_predictions)
                ), 
            "float"
            )
        )
        return tp_op, fp_op, fn_op
