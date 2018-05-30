import tensorflow as tf
import numpy as np

fs = 250 # sampling frequence

# In the original paper, all padding for Conv was 'valid'. 
# And I replace the 'valid' with 'same' for convenience except for conv2d_spatial.
def DeepConvNet(inputs, n_classes, Chans=64, Sample=128, dropoutRate=0.5, training=True, bn_training=True):

    ##############################################################################
    inputs = tf.reshape(inputs, shape=[-1, Chans, Sample, 1], name='Reshape_for_input')
    layer1 = tf.layers.conv2d(inputs, 25, (1, 10), (1, 1), padding='same', activation=None,
                use_bias=True, kernel_initializer=tf.glorot_uniform_initializer(),
                name='conv2d_temporal')                        # output_size [Chans, Sample, 25]
    
    layer1 = tf.layers.conv2d(layer1, 25, (Chans, 1), (1, 1), padding='valid', activation=None,      # output_size [1, Samples, 25]
                use_bias=False, kernel_initializer=tf.glorot_uniform_initializer(),
                name='conv2d_spatial')
    layer1 = tf.layers.batch_normalization(inputs=layer1, axis=-1, training=bn_training, name='bn_1')
    layer1 = tf.nn.elu(layer1, name='elu_1')
    layer1 = tf.layers.average_pooling2d(layer1, (1, 3), (1, 3), name='ave_pooling_1')              # output_size [1, Samples//3, 25]
    layer1 = tf.layers.dropout(layer1, rate=dropoutRate, training=training, name='dropout_1')

    layer2 = tf.layers.conv2d(layer1, 50, (1, 10), (1, 1), padding='same', activation=None,
                use_bias=False, kernel_initializer=tf.glorot_uniform_initializer(),
                name='conv2d_2')
    layer2 = tf.layers.batch_normalization(inputs=layer2, axis=-1, training=bn_training, name='bn_2')
    layer2 = tf.nn.elu(layer2, name='elu_2')
    layer2 = tf.layers.average_pooling2d(layer2, (1, 3), (1, 3), name='ave_pooling_2')              # output_size [1, Samples//9, 50]
    layer2 = tf.layers.dropout(layer2, rate=dropoutRate, training=training, name='dropout_2')


    layer3 = tf.layers.conv2d(layer2, 100, (1, 10), (1, 1), padding='same', activation=None,
                use_bias=False, kernel_initializer=tf.glorot_uniform_initializer(),
                name='conv2d_3')
    layer3 = tf.layers.batch_normalization(inputs=layer3, axis=-1, training=bn_training, name='bn_3')
    layer3 = tf.nn.elu(layer3, name='elu_3')
    layer3 = tf.layers.average_pooling2d(layer3, (1, 3), (1, 3), name='ave_pooling_3')              # output_size [1, Samples//27, 100]
    layer3 = tf.layers.dropout(layer3, rate=dropoutRate, training=training, name='dropout_3')

    layer4 = tf.layers.conv2d(layer3, 200, (1, 10), (1, 1), padding='same', activation=None,
                use_bias=False, kernel_initializer=tf.glorot_uniform_initializer(),
                name='conv2d_4')
    layer4 = tf.layers.batch_normalization(inputs=layer4, axis=-1, training=bn_training, name='bn_4')
    layer4 = tf.nn.elu(layer4, name='elu_4')
    layer4 = tf.layers.average_pooling2d(layer4, (1, 3), (1, 3), name='ave_pooling_4')              # output_size [1, Samples//81, 200]
    
    layer4 = tf.layers.dropout(layer4, rate=dropoutRate, training=training, name='dropout_4')
    
    flatten = tf.reshape(layer4, shape=[-1, (Sample//81)*200])
    output  = tf.layers.dense(flatten, n_classes, name='fc_output')

    return output






class Model(object):
    def __init__(self, batch_size, n_classes=4, dropout=0.5, fs=250, n_Channels=22, n_Samples=500):
        self.n_classes = n_classes
        self.dropout = dropout
        self.fs = fs    # sampling frequence
        self.n_Channels = n_Channels
        self.n_Samples = n_Samples
        self._build_model()
    
    def _build_model(self):
        with tf.name_scope('Inputs'):
            self.X_inputs       = tf.placeholder(tf.float32, [None, self.n_Channels, self.n_Samples, 1], name='X_inputs')
            self.y_inputs       = tf.placeholder(tf.int64, None, name='y_inputs')
            self.tf_is_training = tf.placeholder(tf.bool, None, name='is_training')
            self.tf_bn_training = tf.placeholder(tf.bool, None, name='bn_training')
        
        with tf.variable_scope('EEGNet'):
            self.output = DeepConvNet(self.X_inputs, self.n_classes, self.n_Channels, self.n_Samples, 
                        dropoutRate=self.dropout, training=self.tf_is_training, bn_training=self.tf_bn_training)
        
        with tf.name_scope('Accuracy'):
            prediction = tf.argmax(self.output, axis=1)
            correct_prediction = tf.equal(prediction, self.y_inputs)
            self._acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        with tf.name_scope('Loss'):
            self._loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y_inputs, logits=self.output)