import tensorflow as tf
from spp import spatial_pyramid_pooling
from tensorflow.contrib.layers import flatten

# Enable eager execution to see results without building graph
tf.enable_eager_execution()

class AlexNet_with_spp:

    def __init__(self, input_width, input_height):

        # Local response Normalization constants
        self.lrn_radius = 2
        self.lrn_alpha = 2e-05
        self.lrn_beta = 0.75
        self.lrn_bias = 1.0
        self.dropout_rate = 0.5
        self.num_classes = 1000
        self.spp_bins = [16, 4, 2]
        self.input_width = input_width
        self.input_height = input_height


    def perform_convolution(self, x, filter, out_depth, c_s, k, p_s,
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
                                                 depth_radius=self.lrn_radius,
                                                 alpha=self.lrn_alpha,
                                                 beta=self.lrn_beta,
                                                 bias=self.lrn_bias)

        return conv_out

    def run_feedforward(self, use_spp=True):

        x = tf.random_normal(
                shape=(1, self.input_width, self.input_height, 3))  # (batch_size, width, height, channels)

        # Convolutional Layer 1
        conv_1 = self.perform_convolution(x, 11, 96, 4, 3, 2,
                                     conv_pad="VALID", use_normalization=True)  # (1,96,27,27)

        # Convolutional Layer 2
        conv_2 = self.perform_convolution(conv_1, 5, 256, 1, 3, 2,
                                     conv_pad="SAME", use_normalization=True)  # (1,256,13,13 )

        # Convolutional Layer 3
        conv_3 = self.perform_convolution(conv_2, 3, 384, 1, 3, 2,
                                     conv_pad="SAME", use_max_pooling=False)  # (1,384,13,13)

        # Convolutional Layer 4
        conv_4 = self.perform_convolution(conv_3, 3, 384, 1, 3, 2,
                                     conv_pad="SAME", use_max_pooling=False)  # (1,384,13,13 )

        # Convolutional Layer 5
        conv_5 = self.perform_convolution(conv_4, 3, 256, 1, 3, 2, conv_pad="SAME")  # (1, 256, 6, 6 )

        if use_spp:
            conv_5 = spatial_pyramid_pooling(conv_5, levels=[4, 2, 1])

        # Flatten the neurons
        fc0 = flatten(conv_5)

        # Fully connected layer 1
        fc1_W = tf.truncated_normal(shape=(tf.size(fc0), 4096))
        fc1_b = tf.constant(tf.zeros(4096))
        fc1 = tf.nn.relu(tf.matmul(fc0, fc1_W) + fc1_b)
        fc1 = tf.nn.dropout(fc1, self.dropout_rate)

        # Fully connected layer 2
        fc2_W = tf.truncated_normal(shape=(tf.size(fc1), 4096))
        fc2_b = tf.constant(tf.zeros(4096))
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_W) + fc2_b)
        fc2 = tf.nn.dropout(fc2, self.dropout_rate)

        # Classification Layer
        fc3_W = tf.truncated_normal(shape=(tf.size(fc2), self.num_classes))
        fc3_b = tf.constant(tf.zeros(self.num_classes))
        fc3 = tf.nn.relu(tf.matmul(fc2, fc3_W) + fc3_b)

        # Logits
        logits = tf.nn.softmax(fc3)

        return logits


# Run Alexnet with spp
alexnet = AlexNet_with_spp(input_height=227, input_width=227)
logits = alexnet.run_feedforward()



