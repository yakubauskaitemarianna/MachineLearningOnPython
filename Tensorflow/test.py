"""
Instal tensorflow throught anaconda env:
>> conda install -c anaconda tensorflow
>> conda install -c anaconda tensorflow-gpu
"""

import tensorflow as tf

x1 = tf.constant([1, 2, 3, 4])
x2 = tf.constant([5, 6, 7, 8])

result = tf.multiply(x1, x2)

#print(result)
#Tensor("Mul:0", shape=(4,), dtype=int32)

with tf.Session() as sess:
    output = sess.run(result)
    print(output)

#2019-11-08 17:16:21.572955: I tensorflow/core/platform/cpu_feature_guard.cc:141]
#Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
#[ 5 12 21 32]
