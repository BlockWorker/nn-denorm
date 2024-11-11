import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from fastconv.fastconv import kn2row, fz_arr


class MyConv2D(layers.Conv2D):

    def __init__(
            self,
            filters,
            kernel_size,
            strides=(1, 1),
            padding="same",
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            use_original=False,
            denorm_flush_zero=0,
            **kwargs
    ):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=(1, 1),
            groups=1,
            activation=None,
            data_format=None,
            use_bias=True,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

        self.orig = use_original
        self.flush = denorm_flush_zero

    def convolution_op(self, inputs, kernel):
        if self.orig or tf.is_symbolic_tensor(inputs):
            return super().convolution_op(inputs, kernel)

        if self.data_format != "channels_last":
            inputs = tf.transpose(inputs, perm=[0, 2, 3, 1]).numpy()
        else:
            inputs = inputs.numpy()

        _i = fz_arr(inputs, self.flush)
        _k = fz_arr(kernel.numpy(), self.flush)

        output = kn2row(_i, _k, self.padding, self.strides, flush=self.flush)

        if self.data_format != "channels_last":
            return output.transpose((0, 3, 1, 2))
        else:
            return output

    def call(self, inputs):
        if self.orig or tf.is_symbolic_tensor(inputs):
            return super().call(inputs)
        outputs = self.convolution_op(
            inputs,
            self.kernel,
        )
        if self.use_bias:
            if self.data_format == "channels_last":
                bias_shape = (1,) * (self.rank + 1) + (self.filters,)
            else:
                bias_shape = (1, self.filters) + (1,) * self.rank
            bias = fz_arr(tf.reshape(self.bias, bias_shape).numpy(), self.flush)
            outputs = fz_arr(outputs + bias, self.flush)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs
