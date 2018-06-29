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
                            depthwise_constraint = maxnorm(MAX_NORM, axis=[0,1,2]),
                            use_bias = False)(layer1)
    layer1       = BatchNormalization(axis = 1)(layer1)
    layer1       = Activation('elu')(layer1)                        # output_size [D*F, 1, T]
    layer1       = AveragePooling2D((1, 4))(layer1)                 # output_size [D*F, 1, T//4]
    layer1       = Dropout(dropoutRate)(layer1)         # SpatialDropout2D(dropoutRate)(layer1)
    

    layer2       = SeparableConv2D(filters=numFilters*numSpatialFliters, padding='same',    # equal to DepthwiseConv2D + 1*1-conv2d
                            kernel_size=(1, 16), depth_multiplier=1,
                            depthwise_constraint = maxnorm(MAX_NORM, axis=[0,1,2]),
                            pointwise_initializer = maxnorm(MAX_NORM, axis=[0,1,2]),
                            use_bias = False)(layer1)
    layer2       = BatchNormalization(axis=1)(layer2)
    layer2       = Activation('elu')(layer2)                        # output_size [D*F, 1, T//4]
    layer2       = AveragePooling2D((1, 8))(layer2)
    layer2       = Dropout(dropoutRate)(layer1)         # SpatialDropout2D(dropoutRate)(layer2)            # output_size [D*F, 1, T//32]
    
    
    flatten      = Flatten(name = 'flatten')(layer2)
    
    dense        = Dense(nb_classes, name='dense', kernel_constraint=maxnorm(0.25, axis=0))(flatten)
    softmax      = Activation('softmax', name='softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)


if __name__ == "__main__":
    model = EEGNet_Classifier_new(3)
    model.summary()


# SpatialDropout2D与Dropout的作用类似，但它断开的是整个2D特征图，而不是单个神经元。
# 如果一张特征图的相邻像素之间有很强的相关性（通常发生在低层的卷积层中），
# 那么普通的dropout无法正则化其输出，否则就会导致明显的学习率下降。
# 这种情况下，SpatialDropout2D能够帮助提高特征图之间的独立性，应该用其取代普通的Dropout