import tensorflow as tf
import numpy as np

def spatial_pyramid_pooling(input, levels):
    input_shape = input.get_shape().as_list()
    pyramid = []
    for n in levels:
        stride_1 = np.floor(float(input_shape[1] / n)).astype(np.int32)
        stride_2 = np.floor(float(input_shape[2] / n)).astype(np.int32)
        ksize_1 = stride_1 + (input_shape[1] % n)
        ksize_2 = stride_2 + (input_shape[2] % n)
        pool = tf.nn.max_pool(input,
                              ksize=[1, ksize_1, ksize_2, 1],
                              strides=[1, stride_1, stride_2, 1],
                              padding='VALID')
        pyramid.append(tf.reshape(pool, [input_shape[0], -1]))
    spp_pool = tf.concat(pyramid, axis=1)
    return spp_pool