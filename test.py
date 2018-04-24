import tensorflow as tf
  
A = tf.constant(0.0)
op = tf.assign(A, 10.)
with tf.Session() as sess:  
    sess.run(tf.initialize_all_variables())  
    rs = sess.run(op)
    print(rs)