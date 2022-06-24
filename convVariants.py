"""State of the art convolution layers.

Includes:
    CBAM
    Augmented Attention Convolution layer

Extra notation:
B: batch size
C: channels
H: height
W: width
Nh: number of heads
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models 
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.utils import conv_utils
import tensorflow.keras.activations as activations
import tensorflow.keras.regularizers as regularizers
import tensorflow.keras.initializers as initializers
import tensorflow.keras.constraints as constraints
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.keras import backend as K

# -------------------------------------------------------------------- #
# Helper Functions
# -------------------------------------------------------------------- #

castFloat32 = lambda x: tf.cast(x, dtype=tf.dtypes.float32)

# -------------------------------------------------------------------- #
# Augmented Attention
# -------------------------------------------------------------------- #

class AAConv(layers.Layer):
    """Augmented attention block.

    Shapes:
        INPUT: (B, C, H, W)
        OUPUT: (B, C_2, H, W)

    NOTE: relative positional encoding has not yet been implemented

    Attributes:
        channels_out (int): output channels of this block
        kernel_size (tuple): size of the kernel
        depth_k (int): total depth size for the query or key
        depth_v (int): total depth size for the value
        num_heads (int): number of heads for the attention
        relative_pos (bool): whether to include relative positional encoding
        dilation (int): dilation of the convolution operation
        regularizer (tf.keras regularizer): regularization for the weights and biases
        activation (tf.keras activation): activation function
        kernel_init (function): function for kernel initializer
    """

    def __init__(
        self, 
        channels_out,
        kernel_size,
        depth_k, 
        depth_v, 
        num_heads, 
        relative_pos=False, 
        dilation=1,
        regularizer=None, 
        activation=None,
        kernel_init=None,
        **kwargs):

        super(AAConv, self).__init__(**kwargs)

        if depth_k % num_heads != 0:
            raise ValueError(
                'depth_k {} must be divisible by num_heads {}'.format(
                depth_k, num_heads))

        if depth_v % num_heads != 0:
            raise ValueError(
                'depth_v {} must be divisible by num_heads {}'.format(
                depth_k, num_heads))

        self.channels_out = channels_out
        self.kernel_size = kernel_size
        self.depth_k = depth_k
        self.depth_v = depth_v
        self.num_heads = num_heads
        self.relative_pos = relative_pos
        self.dilation = dilation
        self.regularizer = regularizer
        self.activation = activation
        self.kernel_init = kernel_init

        self.dkh = depth_k // num_heads
        self.dvh = depth_v // num_heads

    def get_config(self):
        config = super().get_config()
        config.update({
            "channels_out":self.channels_out,
            "kernel_size":self.kernel_size,
            "depth_k":self.depth_k,
            "depth_v":self.depth_v,
            "num_heads":self.num_heads,
            "relative_pos":self.relative_pos,
            "dilation":self.dilation,
            "regularizer":self.regularizer,
            "activation":self.activation,
            "kernel_init":self.kernel_init,
        })
        return config


    def build(self, input_shapes):
        
        self.conv = layers.Conv2D(
            self.channels_out - self.depth_v, 
            self.kernel_size,
            data_format='channels_first',
            activation=self.activation,
            kernel_initializer=self.kernel_init,
            kernel_regularizer=self.regularizer,
            bias_regularizer=self.regularizer,
            padding='same',
            dilation_rate=self.dilation,
            name='AA_Conv')

        self.self_atten_conv = layers.Conv2D(
            2 * self.depth_k + self.depth_v, 
            1,
            data_format='channels_first',
            activation=self.activation,
            kernel_initializer=self.kernel_init,
            kernel_regularizer=self.regularizer,
            bias_regularizer=self.regularizer,
            padding='same',
            dilation_rate=self.dilation,
            name='AA_Atten_Conv')

        if self.relative_pos:
            raise NotImplementedError
            self.rel_embed_w = self.add_weight(
                # shape=(2 * , n_in, self.n_out),
                initializer=tf.random_normal_initializer(self.dkh ** -0.5),
                trainable=True,
                regularizer=self.regularizer,
                name='AA_rel_embed_w')


    def _split_heads_2d(self, inputs):
        """Split channels into multiple heads.

        Args:
            inputs: tensor of shape (B, C, H, W)

        Returns:
            tensor of shape (B, Nh, H, W, C // Nh)
        """

        in_shape = tf.shape(inputs)

        ret_shape = [
            in_shape[0], 
            self.num_heads, 
            in_shape[1] // self.num_heads, 
            in_shape[2],
            in_shape[3]]

        # (B, Nh, C // Nh, H, W)
        split = tf.reshape(inputs, ret_shape)

        # (B, Nh, H, W, C // Nh)
        result = tf.transpose(split, [0, 1, 3, 4, 2])

        return result

    
    def _combine_heads_2d(self, inputs):
        """Combine the heads together.

        Args:
            tensor of shape (B, Nh, H, W, C)  

        Returns:
            tensor of shape (B, H, W, Nh * C)  
        """

        # (B, H, W, NUM_HEADS, C_IN)  
        transposed = tf.transpose(inputs, [0, 2, 3, 1, 4])

        trans_shape = tf.shape(transposed)
        # N_H, C = tf.shape(transposed)[-2:]

        ret_shape = tf.concat(
            [tf.shape(transposed)[:-2], 
            [trans_shape[-1] * trans_shape[-2]]], 
            axis=0)
        result = tf.reshape(transposed, ret_shape)

        return result


    def _relative_logits(self, inputs, height, width):
        """Compute relative logits
        """

        # relative logits in width dimension
        raise NotImplementedError(
            "Relative positional encoding has not yet been implemented.")

            
    def _self_attention_2d(self, inputs):
        """Apply self 2d self attention to input.

        NOTE: unlike the implementation in the paper, we do 
        not have an extra convolution layer for projection at
        the end of the self attention
        
        Args:
            inputs: tensor of shape (B, C, H, W)

        Returns:
            tensor of shape (B, depth_v, H, W)  
        """

        in_shape = tf.shape(inputs)
        H = in_shape[2]
        W = in_shape[3]

        kqv = self.self_atten_conv(inputs)

        # (B, dk or dv, H, W)
        k, q, v = tf.split(
            kqv, 
            [self.depth_k, self.depth_k, self.depth_v], 
            axis=1)

        q *= self.dkh ** -0.5 # scaled dot−product

        # (B, Nh, H, W, dk or dv // Nh)
        q = self._split_heads_2d(q)
        k = self._split_heads_2d(k)
        v = self._split_heads_2d(v)


        # returns shape: (B, NUM_HEADS, H * W, d)
        flatten_hw = lambda x, d: tf.reshape(x, [-1, self.num_heads, H * W, d])

        # (B, NUM_HEADS, H * W, H * W)
        logits = tf.linalg.matmul(
            flatten_hw(q, self.dkh),
            flatten_hw(k, self.dkh),
            transpose_b=True)

        if self.relative_pos:
            rel_logits_h, rel_logits_w = self._relative_logits(q, H, W)
            logits += rel_logits_h
            logits += rel_logits_w

        weights = tf.math.softmax(logits)

        # (B, NUM_HEADS, H * W, dvh)
        attn_out = tf.linalg.matmul(
            weights, 
            flatten_hw(v, self.dvh))

        # (B, NUM_HEADS, H, W, dvh)    
        attn_out = tf.reshape(
            attn_out, 
            [-1, self.num_heads, H, W, self.dvh]) 

        # (B, H, W, NUM_HEADS * dvh)   
        attn_out = self._combine_heads_2d(attn_out)  

        # (B, NUM_HEADS * dvh = dv, H, W)   
        attn_out = tf.transpose(attn_out, [0, 3, 1, 2])

        return attn_out


    @tf.function
    def call(self, inputs):
        conv_out = self.conv(inputs)
        attn_out = self._self_attention_2d(inputs)
        result = tf.concat([conv_out, attn_out], axis=1)

        return result

# -------------------------------------------------------------------- #
# CBAM
# -------------------------------------------------------------------- #

class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, filters, ratio,**kwargs):
        super(ChannelAttention, self).__init__()
        self.filters = filters
        self.ratio = ratio
        
    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({
            "filters": self.filters,
            "ratio": self.ratio,
            })
        return config
    
    def build(self, input_shape):
        self.shared_layer_one = tf.keras.layers.Dense(self.filters//self.ratio,
                                                        activation='relu', kernel_initializer='he_normal',
                                                        use_bias=True,
                                                        bias_initializer='zeros')
        self.shared_layer_two = tf.keras.layers.Dense(self.filters,
                                                        kernel_initializer='he_normal',
                                                        use_bias=True,
                                                        bias_initializer='zeros')
        
    def call(self, inputs):
        # AvgPool
        avg_pool = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)

        # MaxPool
        max_pool = tf.keras.layers.GlobalMaxPooling2D()(inputs)
        max_pool = tf.keras.layers.Reshape((1,1,self.filters))(max_pool)
        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)

        attention = tf.keras.layers.Add()([avg_pool,max_pool])
        attention = tf.keras.layers.Activation('sigmoid')(attention)

        return tf.keras.layers.Multiply()([inputs, attention])

class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, kernel_size, **kwargs):
        super().__init__()
        self.kernel_size = kernel_size
        
    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        config.update({
            "kernel_size": self.kernel_size,
        })
        return config
        
    def build(self, input_shape):
        self.conv2d = tf.keras.layers.Conv2D(filters = 1,
                kernel_size=self.kernel_size,
                strides=1,
                padding='same',
                activation='sigmoid',
                kernel_initializer='he_normal',
                use_bias=False)

    def call(self, inputs):
        
        # AvgPool
        avg_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(inputs)
        
        # MaxPool
        max_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(inputs)
        attention = tf.keras.layers.Concatenate(axis=3)([avg_pool, max_pool])
        attention = self.conv2d(attention)

        return tf.keras.layers.multiply([inputs, attention]) 

# -------------------------------------------------------------------- #
# Grouped Convolution
# -------------------------------------------------------------------- #


class GroupConvBase(tf.keras.layers.Layer):
    def __init__(
        self, 
        rank, 
        filters, 
        kernel_size, 
        groups=1, 
        strides=1, 
        padding='VALID', 
        data_format=None,
        dilation_rate=1,
        activation=None, 
        use_bias=True, 
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros', 
        kernel_regularizer=None, 
        bias_regularizer=None, 
        activity_regularizer=None,
        kernel_constraint=None, 
        bias_constraint=None, **kwargs):

        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        if filters % groups != 0:
            raise ValueError("Groups must divide filters evenly, but got {}/{}".format(filters, groups))

        self.filters = filters
        self.groups = groups
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.data_format = data_format
        self.padding = padding
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.rank = rank

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if conv_utils.normalize_data_format(self.data_format) == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        kernel_shape = self.kernel_size + (input_dim // self.groups, self.filters)

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None

        self.built = True

    def call(self, inputs):
        outputs = tf.nn.conv2d(
            inputs, 
            self.kernel, 
            strides=self.strides,
            data_format="NCHW", 
            dilations=self.dilation_rate,
            name=self.name,
            padding=self.padding)

        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                    outputs += bias
                else:
                    outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            "groups": self.groups,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        return {list(super(GroupConvBase, self).get_config().items()) + list(config.items())}


class GroupConv2D(GroupConvBase):
    """Grouped convolution. 
    
    Essentially the same as regular Conv2D, except inputs are seperated int
    groups.
    Code is taken from here: https://github.com/tensorflow/tensorflow/issues/34024
    Also, check out this pull request: https://github.com/tensorflow/tensorflow/pull/25818
    Attributes:
        groups (int): number of groups to split the convolution layer.
    """

    def __init__(
        self, 
        filters, 
        kernel_size,
        groups, 
        strides=(1, 1), 
        padding='valid', 
        data_format=None, 
        dilation_rate=(1, 1), 
        activation=None,
        use_bias=True, 
        kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', 
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None, 
        kernel_constraint=None, 
        bias_constraint=None,
        **kwargs):

        super(GroupConv2D, self).__init__(
            rank=2,
            filters=filters, 
            kernel_size=kernel_size, 
            groups=groups, 
            strides=strides,
            padding=padding.upper(),
            data_format=data_format, dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias, kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)

# -------------------------------------------------------------------- #
# DropBlock
# -------------------------------------------------------------------- #

def _bernoulli(shape, mean):
    return tf.nn.relu(tf.sign(mean - tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.float32)))

class DropBlock2D(tf.keras.layers.Layer):
    def __init__(self, keep_prob, block_size, scale=True, **kwargs):
        super(DropBlock2D, self).__init__(**kwargs)
        self.keep_prob = float(keep_prob) if isinstance(keep_prob, int) else keep_prob
        self.block_size = int(block_size)
        self.scale = tf.constant(scale, dtype=tf.bool) if isinstance(scale, bool) else scale

    def get_config(self):
        config = super().get_config()
        config.update({
            "keep_prob":self.keep_prob,
            "block_size":self.block_size,
        })
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        assert len(input_shape) == 4
        _, self.h, self.w, self.channel = input_shape.as_list()
        # pad the mask
        p1 = (self.block_size - 1) // 2
        p0 = (self.block_size - 1) - p1
        self.padding = [[0, 0], [p0, p1], [p0, p1], [0, 0]]
        self.set_keep_prob()
        super(DropBlock2D, self).build(input_shape)

    def call(self, inputs, training=True, **kwargs):
        def drop():
            mask = self._create_mask(tf.shape(inputs))
            output = inputs * mask
            output = tf.cond(self.scale,
                             true_fn=lambda: output * tf.cast(tf.size(mask), dtype=tf.float32) / tf.reduce_sum(mask),
                             false_fn=lambda: output)
            return output

        if training is None:
            training = K.learning_phase()
        output = tf.cond(tf.logical_or(tf.logical_not(training), tf.equal(self.keep_prob, 1.0)),
                         true_fn=lambda: inputs,
                         false_fn=drop)
        return output

    def set_keep_prob(self, keep_prob=None):
        """This method only supports Eager Execution"""
        if keep_prob is not None:
            self.keep_prob = keep_prob
        w, h = tf.cast(self.w, dtype=tf.float32), tf.cast(self.h, dtype=tf.float32)
        self.gamma = (1. - self.keep_prob) * (w * h) / (self.block_size ** 2) / \
                     ((w - self.block_size + 1) * (h - self.block_size + 1))

    def _create_mask(self, input_shape):
        sampling_mask_shape = tf.stack([input_shape[0],
                                       self.h - self.block_size + 1,
                                       self.w - self.block_size + 1,
                                       self.channel])
        mask = _bernoulli(sampling_mask_shape, self.gamma)
        mask = tf.pad(mask, self.padding)
        mask = tf.nn.max_pool(mask, [1, self.block_size, self.block_size, 1], [1, 1, 1, 1], 'SAME')
        mask = 1 - mask
        return mask
