import tensorflow as tf
import numpy as np


def my_conv2d(inputs, filters, kernel_size, strides=(1, 1), padding='same', activation=None, name=None, reuse=None):
    return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), bias_initializer=tf.constant_initializer(0.1), name=name, reuse=reuse)
                # tf.glorot_uniform_initializer()   # Xavier初始化方法，还有normal形式，但貌似uniform用的比较多

def my_conv2d_bn(inputs, filters, kernel_size, strides=(1, 1), padding='same', activation=None, name=None, reuse=None, training=True):
    with tf.variable_scope(name, reuse=reuse):
        conv = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=None, use_bias=False,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), bias_initializer=tf.constant_initializer(0.1))
        conv_bn = tf.layers.batch_normalization(inputs=conv, axis=-1, momentum=0.99, epsilon=1e-8, training=training)
        output = conv_bn if activation is None else activation(conv_bn)

    return output


def max_norm_regularizer(threshold, axes=1, name='max_nrom', collection='max-norm'):
    def max_norm(weights):
        clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
        clip_weithts = tf.assign(weights, clipped, name=name)
        tf.add_to_collection(collection, clip_weithts)
        return None
    return max_norm

'''
max_norm_reg = max_norm_regularizer(threshold=1.0)

clip_all_weights = tf.get_collection('max-norm')

sess.run([trian_op,..], feed_dict={...})
sess.run(clip_all_weights) is needed
'''
'''
all_vars= tf.global_variables()
def get_var(name):
    for i in range(len(all_vars)):
        if all_vars[i].name.startswith(name):
            return all_vars[i]
    return None
fc1_var = get_var('LogReg/fc1/weights')
kernel = tf.get_collection(tf.GraphKeys.VARIABLES, 'conv1/kernel')[0]
'''

fs = 250 # sampling frequence


def EEGNet_Classifier(inputs, n_classes, Chans=64, Sample=128, dropoutRate=0.25,
        kernLength=fs//2, F1=4, D=2, F2=8, training=True, bn_training=True):
        # kernLength should be half of the fs
    MAX_NORM = 1.0
    max_norm_reg = max_norm_regularizer(threshold=MAX_NORM, axes=3)
    Weight_depthwise = tf.get_variable('Weight_depthwise', [Chans, 1, F1, D],
            initializer=tf.glorot_uniform_initializer(), regularizer=max_norm_reg)
    max_norm_reg(Weight_depthwise)

    ##################################################################
    inputs = tf.reshape(inputs, shape=[-1, Chans, Sample, 1], name='Reshape_for_input') # for 2D_conv channels_last
    
    layer1 = tf.layers.conv2d(inputs, F1, (1, kernLength), (1, 1), padding='same',
                activation=None, use_bias=False, kernel_initializer=tf.glorot_uniform_initializer(),
                name='conv2d_1')
                # kernel_regularizer=max_norm_reg, name='conv2d_1')
    layer1 = tf.layers.batch_normalization(inputs=layer1, axis=-1, training=bn_training)           # output_size [C, T, F1]
                                                                                                # C为电极通道数，T为采样点数，F1为filters通道数
    layer1 = tf.nn.depthwise_conv2d(input=layer1, filter=Weight_depthwise, # in the tf.nn, the flilter must be a tensor
                strides=[1, 1, 1, 1], padding='VALID', name='depthwise_conv2d_1')               # output_size [1, T, F1*D] 由于后面有BN，因此不用加偏置
    layer1 = tf.layers.batch_normalization(inputs=layer1, axis=-1, training=training)
    layer1 = tf.nn.elu(layer1)
    layer1 = tf.layers.average_pooling2d(layer1, (1, 4), strides=(1, 4), name='ave_pooling')    # output_size [1, T//4, F1*D]
    layer1 = tf.layers.dropout(layer1, rate=dropoutRate, training=training, name='dropout_1')



    layer2 = tf.layers.separable_conv2d(layer1, filters=F2,
                kernel_size=(1, 16), strides=(1,1), padding='same',
                depth_multiplier=1, activation=None, use_bias=False,
                depthwise_initializer=tf.glorot_uniform_initializer(),
                pointwise_initializer=tf.glorot_uniform_initializer(),
                # depthwise_regularizer=max_norm_reg, pointwise_regularizer=max_norm_reg,
                name='separable_conv2d')
    layer2 = tf.layers.batch_normalization(inputs=layer2, axis=-1, training=bn_training)
    layer2 = tf.nn.elu(layer2)                                                                  # output_size [1, T//4, F2]
    layer2 = tf.layers.average_pooling2d(layer2, (1, 8), strides=(1, 8), name='ave_pooling2')   # output_size [1, T//32, F2]
    layer2 = tf.layers.dropout(layer2, rate=dropoutRate, training=training, name='dropout_2')
    
    flatten = tf.reshape(layer2, shape=[-1, (Sample//32)*F2])
    output  = tf.layers.dense(flatten, n_classes, 
                kernel_regularizer=max_norm_regularizer(threshold=0.25, axes=1), name='output')

    return output



class Model(object):
    def __init__(self, batch_size, n_classes=4, dropout=0.25, fs=250, n_Channels=22, n_Samples=500):
        self.n_classes = n_classes
        self.dropout = dropout
        self.fs = fs    # sampling frequence
        self.n_Channels = n_Channels
        self.n_Samples = n_Samples
        self._build_model()
    
    def _build_model(self):
        with tf.name_scope('Inputs'):
            self.X_inputs       = tf.placeholder(tf.float32, [None, self.n_Channels, self.n_Samples, 1], name='X_inputs')
            self.y_inputs       = tf.placeholder(tf.int64, [None, 1], name='y_inputs')
            self.tf_is_training = tf.placeholder(tf.bool, None, name='is_training')
            self.tf_bn_training = tf.placeholder(tf.bool, None, name='bn_training')
        
        with tf.name_scope('EEGNet'):
            self.output = EEGNet_Classifier(self.X_inputs, self.n_classes, self.n_Channels, self.n_Samples, 
                        dropoutRate=self.dropout, kernLength=self.fs//2, training=self.tf_is_training,
                        bn_training=self.tf_bn_training)
        
        with tf.name_scope('Accuracy'):
            prediction = tf.argmax(self.output, axis=1)
            correct_prediction = tf.equal(prediction, self.y_inputs)
            self._acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        with tf.name_scope('Loss'):
            self._loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y_inputs, logits=self.output)