import tensorflow as tf
from spp import spatial_pyramid_pool

# Enable eager execution to see results without building graph
tf.enable_eager_execution()

# Local response Normalization constants
_RADIUS = 2
_ALPHA = 2e-05
_BETA = 0.75
_BIAS = 1.0

x = tf.random_normal(
    shape=(1, 227, 227, 3))  # (batch_size, width, height, channels)

def perform_convolution(x, filter, out_depth, c_s, k, p_s,
                        conv_pad, pooling_pad="VALID", use_normalization=False, use_max_pooling=True):
    """
    :param x: tf tensor -> incoming tensor
    :param filter: int --> filter shape for convolution
    :param out_depth: int - expected depth of layer
    :param c_s: int - convolution stride ( same in x and y )
    :param p_s:int - pooling stride ( same in x and y )
    :param k : int - kernel size for max pooling ( same in x and y )
    :param padding: str -> "Valid" or "Same"
    :param use_normalization: boolean -> whether to use normalization
    :param use_max_pooling: boolean -> whether to use normalization
    :return: convolutional output of current layer
    """
    conv_W = tf.truncated_normal(shape=(filter, filter, tf.shape(x)[3], out_depth))
    conv_b = tf.constant(tf.zeros(out_depth))
    conv_in = tf.nn.conv2d(x, conv_W, strides=[1, c_s, c_s, 1],
                            padding=conv_pad) + conv_b
    conv_out = tf.nn.relu(conv_in)

    if use_max_pooling:
        conv_out = tf.nn.max_pool(conv_out, ksize=[1, k, k, 1],
                                 strides=[1, p_s, p_s, 1], padding=pooling_pad)

    if use_normalization:
        conv_out = tf.nn.local_response_normalization(conv_out,
                                             depth_radius=_RADIUS,
                                             alpha=_ALPHA,
                                             beta=_BETA,
                                             bias=_BIAS)

    return conv_out


# Convolutional Layer 1
conv_1 = perform_convolution(x, 11, 96, 4, 3, 2,
                             conv_pad="VALID", use_normalization=True)  # (1,96,27,27)

# Convolutional Layer 2
conv_2 = perform_convolution(conv_1, 5, 256, 1, 3, 2,
                             conv_pad="SAME", use_normalization=True)  # (1,256,13,13 )

# Convolutional Layer 3
conv_3 = perform_convolution(conv_2, 3, 384, 1, 3, 2,
                             conv_pad="SAME", use_max_pooling=False)  # (1,384,13,13)

# Convolutional Layer 4
conv_4 = perform_convolution(conv_3, 3, 384, 1, 3, 2,
                             conv_pad="SAME", use_max_pooling=False)  # (1,384,13,13 )

# Convolutional Layer 5
conv_5 = perform_convolution(conv_4, 3, 256, 1, 3, 2, conv_pad="SAME")  #(1, 256, 6, 6 )
print("done")
