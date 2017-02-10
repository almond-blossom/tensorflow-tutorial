
import tensorflow as tf
import numpy as np

char_rdic = ['h', 'e', 'l', 'o']
char_dic = {w: i for i, w in enumerate(char_rdic)}
x_data = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 1, 0]],
                  dtype='f')

sample = [char_dic[c] for c in "hello"]

char_vocab_size = len(char_dic)
rnn_size = char_vocab_size
time_step_size = 4
batch_size = 1

rnn_cell = tf.nn.rnn_cell.BasicRNNCell(rnn_size)
state = tf.zeros([batch_size, rnn_cell.state_size])
X_split = tf.split(0, time_step_size, x_data)
outputs, state = tf.nn.rnn(rnn_cell, X_split, state)

logits = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
targets = tf.reshape(sample[1:], [-1])
weights = tf.ones([time_step_size * batch_size])

loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [weights])
cost = tf.reduce_sum(loss) / batch_size
train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(100):
        sess.run(train_op)
        result = sess.run(tf.arg_max(logits, 1))
        print(result, [char_rdic[t] for t in result])
