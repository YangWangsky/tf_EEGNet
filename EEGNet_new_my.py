import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D
from keras.layers.convolutional import SeparableConv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import SpatialDropout2D
from keras.regularizers import l1_l2
from keras.constraints import maxnorm
from keras.layers import Input, Flatten
from keras.applications.mobilenet import DepthwiseConv2D

"""MaxNorm weight constraint.

Constrains the weights incident to each hidden unit
to have a norm less than or equal to a desired value.

# Arguments
    m: the maximum norm for the incoming weights.
    axis: integer, axis along which to calculate weight norms.
        For instance, in a `Dense` layer the weight matrix
        has shape `(input_dim, output_dim)`,
        set `axis` to `0` to constrain each weight vector
        of length `(input_dim,)`.
        In a `Conv2D` layer with `data_format="channels_last"`,
        the weight tensor has shape
        `(rows, cols, input_depth, output_depth)`,
        set `axis` to `[0, 1, 2]`
        to constrain the weights of each filter tensor of size
        `(rows, cols, input_depth)`.
"""

def EEGNet_Classifier(nb_classes, Chans = 64, Samples = 128, regRate = 0.0001,
           dropoutRate = 0.25, kernLength = 64, numFilters = 8):
    """ Keras Implementation of EEGNet (https://arxiv.org/abs/1611.08024v3)

    Requires Tensorflow >= 1.5 and Keras >= 2.1.3
    
    Note that we use 'image_data_format' = 'channels_first' in there keras.json
    configuration file.

    Inputs:
        
        nb_classes: int, number of classes to classify
        Chans, Samples: number of channels and time points in the EEG data
        regRate: regularization parameter for L1 and L2 penalties
        dropoutRate: dropout fraction
        kernLength: length of temporal convolution in first layer
        numFilters: number of temporal-spatial filter pairs to learn

    Depending on the task, using numFilters = 4 or 8 seemed to do pretty well
    across tasks. 
    
    """

    input1 = Input(shape=(1, Chans, Samples))

    ##################################################################
    layer1       = Conv2D(numFilters, (1, kernLength), padding = 'same',
                          kernel_regularizer = l1_l2(l1=0.0, l2=0.0),
                          input_shape = (1, Chans, Samples),
                          use_bias = False)(input1)
    layer1       = BatchNormalization(axis = 1)(layer1)
    layer1       = DepthwiseConv2D((Chans, 1), 
                              depthwise_regularizer = l1_l2(l1=regRate, l2=regRate),
                              use_bias = False)(layer1)
    layer1       = BatchNormalization(axis = 1)(layer1)
    layer1       = Activation('elu')(layer1)
    layer1       = SpatialDropout2D(dropoutRate)(layer1)
    
    layer2       = SeparableConv2D(numFilters, (1, 8), 
                              depthwise_regularizer=l1_l2(l1=0.0, l2=regRate),
                              use_bias = False, padding = 'same')(layer1)
    layer2       = BatchNormalization(axis=1)(layer2)
    layer2       = Activation('elu')(layer2)
    layer2       = AveragePooling2D((1, 4))(layer2)
    layer2       = SpatialDropout2D(dropoutRate)(layer2)
    
    layer3       = SeparableConv2D(numFilters*2, (1, 8), depth_multiplier = 2,
                              depthwise_regularizer=l1_l2(l1=0.0, l2=regRate), 
                              use_bias = False, padding = 'same')(layer2)
    layer3       = BatchNormalization(axis=1)(layer3)
    layer3       = Activation('elu')(layer3)
    layer3       = AveragePooling2D((1, 4))(layer3)
    layer3       = SpatialDropout2D(dropoutRate)(layer3)
    
    
    flatten      = Flatten(name = 'flatten')(layer3)
    
    dense        = Dense(nb_classes, name = 'dense')(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)


# 将原有的l1_l2正则化替换成了max_norm
# 增加DepthwiseConv2D的空间滤波的通道数
# 减少了一层SeparableConv2D
MAX_NORM = 1.0
fs = 250        # sampling frequency    
# Setting the length of the temporal kernel at half the sampling rate allows for capturing frequency information at 2Hz and above
def EEGNet_Classifier_new(nb_classes, Chans = 64, Samples = 128, regRate = 0.0001,
           dropoutRate = 0.25, kernLength = fs//2, numFilters = 8, numSpatialFliters=1):
    # kernlenth的取值为采样频率的一半

    input1 = Input(shape=(1, Chans, Samples))

    ##################################################################
    layer1       = Conv2D(numFilters, (1, kernLength), padding='same',            # temporal kernel
                            kernel_constraint = maxnorm(MAX_NORM, axis=[1,2,3]),
                            input_shape = (1, Chans, Samples),      # channels_first 
                            use_bias = False)(input1)               # output_size [F, C, T]
    layer1       = BatchNormalization(axis = 1)(layer1)             # bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
    layer1       = DepthwiseConv2D((Chans, 1), padding='valid',                     # spatial filters within each feature map
                            depth_multiplier=numSpatialFliters,
                            depthwise_constraint = maxnorm(MAX_NORM, axis=[1,2,3]),
                            use_bias = False)(layer1)
    layer1       = BatchNormalization(axis = 1)(layer1)
    layer1       = Activation('elu')(layer1)                        # output_size [D*F, 1, T]
    layer1       = AveragePooling2D((1, 4))(layer1)                 # output_size [D*F, 1, T//4]
    layer1       = Dropout(dropoutRate)(layer1)         # SpatialDropout2D(dropoutRate)(layer1)
    

    layer2       = SeparableConv2D(filters=numFilters*numSpatialFliters, padding='same',    # equal to DepthwiseConv2D + 1*1-conv2d
                            kernel_size=(1, 16), depth_multiplier=1,
                            depthwise_constraint = maxnorm(MAX_NORM, axis=[1,2,3]),
                            pointwise_initializer = maxnorm(MAX_NORM, axis=[1,2,3]),
                            use_bias = False)(layer1)
    layer2       = BatchNormalization(axis=1)(layer2)
    layer2       = Activation('elu')(layer2)                        # output_size [D*F, 1, T//4]
    layer2       = AveragePooling2D((1, 8))(layer2)
    layer2       = Dropout(dropoutRate)(layer1)         # SpatialDropout2D(dropoutRate)(layer2)            # output_size [D*F, 1, T//32]
    
    
    flatten      = Flatten(name = 'flatten')(layer2)
    
    dense        = Dense(nb_classes, name='dense', kernel_constraint=maxnorm(0.25, axis=0))(flatten)
    softmax      = Activation('softmax', name='softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)


def EEGNet_regression(activation, Chans = 64, Samples = 128, regRate = 0.0001,
           dropoutRate = 0.25, kernLength = fs//2, numFilters = 8, numSpatialFliters=1):
    # kernlenth的取值为采样频率的一半

    input1 = Input(shape=(1, Chans, Samples))

    ##################################################################
    layer1       = Conv2D(numFilters, (1, kernLength), padding='same',            # temporal kernel
                            kernel_constraint = maxnorm(MAX_NORM, axis=[1,2,3]),
                            input_shape = (1, Chans, Samples),      # channels_first 
                            use_bias = False)(input1)               # output_size [F, C, T]
    layer1       = BatchNormalization(axis = 1)(layer1)             # bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
    layer1       = DepthwiseConv2D((Chans, 1), padding='valid',                     # spatial filters within each feature map
                            depth_multiplier=numSpatialFliters,
                            depthwise_constraint = maxnorm(MAX_NORM, axis=[1,2,3]),
                            use_bias = False)(layer1)
    layer1       = BatchNormalization(axis = 1)(layer1)
    layer1       = Activation('elu')(layer1)                        # output_size [D*F, 1, T]
    layer1       = AveragePooling2D((1, 4))(layer1)                 # output_size [D*F, 1, T//4]
    layer1       = Dropout(dropoutRate)(layer1)         # SpatialDropout2D(dropoutRate)(layer1)
    

    layer2       = SeparableConv2D(filters=numFilters*numSpatialFliters, padding='same',    # equal to DepthwiseConv2D + 1*1-conv2d
                            kernel_size=(1, 16), depth_multiplier=1,
                            depthwise_constraint = maxnorm(MAX_NORM, axis=[1,2,3]),
                            pointwise_initializer = maxnorm(MAX_NORM, axis=[1,2,3]),
                            use_bias = False)(layer1)
    layer2       = BatchNormalization(axis=1)(layer2)
    layer2       = Activation('elu')(layer2)                        # output_size [D*F, 1, T//4]
    layer2       = AveragePooling2D((1, 8))(layer2)
    layer2       = Dropout(dropoutRate)(layer1)         # SpatialDropout2D(dropoutRate)(layer2)            # output_size [D*F, 1, T//32]
    
    
    flatten     = Flatten(name = 'flatten')(layer2)
    dense       = Dense(1, name='dense', kernel_constraint=maxnorm(0.25, axis=0))(flatten)
    output      = Activation(activation)(dense)
    
    return Model(inputs=input1, outputs=output)

if __name__ == "__main__":
    model = EEGNet_Classifier_new(3)
    model.summary()


# SpatialDropout2D与Dropout的作用类似，但它断开的是整个2D特征图，而不是单个神经元。
# 如果一张特征图的相邻像素之间有很强的相关性（通常发生在低层的卷积层中），
# 那么普通的dropout无法正则化其输出，否则就会导致明显的学习率下降。
# 这种情况下，SpatialDropout2D能够帮助提高特征图之间的独立性，应该用其取代普通的Dropout