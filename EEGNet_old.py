from keras.models import Model
from keras.layers.core import Dense, Activation
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.layers.convolutional import SeparableConv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import SpatialDropout2D
from keras.regularizers import l1_l2
from keras.layers import Input, Flatten
from keras.applications.mobilenet import DepthwiseConv2D

# 跟原文的改变应该挺大的，有时间看一下论文


def EEGNet(nb_classes, Chans = 64, Samples = 128, regRate = 0.0001,
           dropoutRate = 0.25, kernLength = 64, numFilters = 8):
    """ Keras Implementation of EEGNet (https://arxiv.org/abs/1611.08024v3)

    Requires Tensorflow >= 1.5 and Keras >= 2.1.3

    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. 	
    
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

    input1   = Input(shape = (1, Chans, Samples))

    ##################################################################
    layer1       = Conv2D(numFilters, (1, kernLength), padding = 'same',
                          kernel_regularizer = l1_l2(l1=0.0, l2=0.0),
                          input_shape = (1, Chans, Samples),
                          use_bias = False)(input1)
    layer1       = BatchNormalization(axis = 1)(layer1)
    layer1       = DepthwiseConv2D((Chans, 1),                  # out shape: [batch, out_height, out_width, in_channels * channel_multiplier]
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
                                                                                    # SeparableConv2D等价于先做DepthwiseConv2D，再进行1*1的卷积？
    layer3       = SeparableConv2D(numFilters*2, (1, 8), depth_multiplier = 2,      # [1, 1, channel_multiplier * in_channels, out_channels]
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

