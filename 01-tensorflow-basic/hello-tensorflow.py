
import tensorflow as tf

hello = tf.constant('Hello, Tensorflow')

print(hello)

sess = tf.Session()

print(sess.run(hello))
