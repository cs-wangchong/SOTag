# -*- coding: utf-8 -*-

import tensorflow as tf


class TextCNN:
    def __init__(self, filter_sizes, num_filters, k, vocab_size, num_classes, learning_rate, decay_steps, decay_rate, sequence_length, embed_size,
                 is_training, initializer=tf.random_normal_initializer(stddev=0.1), clip_gradients=5.0, decay_rate_big=0.50):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.filter_sizes = filter_sizes  # it is a list of int. e.g. [3,4,5]
        self.num_filters = num_filters
        self.initializer = initializer
        self.num_filters_total = self.num_filters * len(filter_sizes)  # how many filters totally.
        self.clip_gradients = clip_gradients

        # add placeholder (X,label)
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  # X
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
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
        self.loss_val = self.loss()
        self.train_op = self.train()

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding"):  # embedding matrix
            # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size], initializer=self.initializer)
            self.W_projection = tf.get_variable("W_projection", shape=[self.num_filters_total,
                                                                       self.num_classes], initializer=self.initializer)  # [embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])  # [label_size] #ADD 2017.06.09

    def inference(self):
        """main computation graph here: 1.embedding-->2.average-->3.linear classifier"""
        # 1.=====>get emebedding of words in the sentence
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)  # [None,sentence_length,embed_size]
        # [None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv
        self.sentence_embeddings_expanded = tf.expand_dims(self.embedded_words, -1)

        # 2.=====>loop each filter size. for each filter, do:convolution-pooling layer(a.create filters,b.conv,c.apply nolinearity,d.max-pooling)--->
        # you can use:tf.nn.conv2d;tf.nn.relu;tf.nn.max_pool; feature shape is 4-d. feature is a new variable
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("convolution-pooling-%s" % filter_size):
                # ====>a.create filter
                filter = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 1, self.num_filters], initializer=self.initializer)
                # ====>b.conv operation: conv2d===>computes a 2-D convolution given 4-D `input` and `filter` tensors.
                # Conv.Input: given an input tensor of shape `[batch, in_height, in_width, in_channels]` and a filter / kernel tensor of shape `[filter_height, filter_width, in_channels, out_channels]`
                # Conv.Returns: A `Tensor`. Has the same type as `input`.
                #         A 4-D tensor. The dimension order is determined by the value of `data_format`, see below for details.
                # 1)each filter with conv2d's output a shape:[1,sequence_length-filter_size+1,1,1];2)*num_filters--->[1,sequence_length-filter_size+1,1,num_filters];3)*batch_size--->[batch_size,sequence_length-filter_size+1,1,num_filters]
                # input data format:NHWC:[batch, height, width, channels];output:4-D
                # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                # ====>c. apply nolinearity
                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])  # ADD 2017-06-09
                # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`
                h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")
                # ====>. max-pooling.  value: A 4-D `Tensor` with shape `[batch, height, width, channels]
                #                  ksize: A list of ints that has length >= 4.  The size of the window for each dimension of the input tensor.
                #                  strides: A list of ints that has length >= 4.  The stride of the sliding window for each dimension of the input tensor.
                # shape:[batch_size, 1, 1, num_filters].max_pool:performs the max pooling on the input.
                pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)
        # 3.=====>combine all pooled features, and flatten the feature.output' shape is a [1,None]
        # e.g. >>> x1=tf.ones([3,3]);x2=tf.ones([3,3]);x=[x1,x2]
        #         x12_0=tf.concat(x,0)---->x12_0' shape:[6,3]
        #         x12_1=tf.concat(x,1)---->x12_1' shape;[3,6]
        # shape:[batch_size, 1, 1, num_filters_total]. tf.concat=>concatenates tensors along one dimension.where num_filters_total=num_filters_1+num_filters_2+num_filters_3
        self.h_pool = tf.concat(pooled_outputs, 3)
        # shape should be:[None,num_filters_total]. here this operation has some result as tf.sequeeze().e.g. x's shape:[3,3];tf.reshape(-1,x) & (3, 3)---->(1,9)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])

        # 4.=====>add dropout: use tf.nn.dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob=self.dropout_keep_prob)  # [None,num_filters_total]

        # 5. logits(use linear layer)and predictions(argmax)
        with tf.name_scope("output"):
            # shape:[None, self.num_classes]==tf.matmul([None,self.embed_size],[self.embed_size,self.num_classes])
            logits = tf.matmul(self.h_drop, self.W_projection) + self.b_projection
        return logits

    def loss(self, l2_lambda=0.00001):  # 0.0001
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

    def train(self):
        """based on the loss, use SGD to update parameter"""
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=self.lr, optimizer="Adam", clip_gradients=self.clip_gradients)
        return train_op
