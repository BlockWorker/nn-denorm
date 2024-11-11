import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from fastconv.fastconv import tiled_matmul, fz_arr


class MyDense(layers.Dense):
    def __init__(
            self,
            units,
            activation=None,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            use_original=False,
            denorm_flush_zero=0,
            **kwargs,
    ):
        super().__init__(
            units,
            activation,
            use_bias,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
            **kwargs,
        )

        self.orig = use_original
        self.flush = denorm_flush_zero

    def call(self, inputs):
        if self.orig or tf.is_symbolic_tensor(inputs):
            return super().call(inputs)

        i = fz_arr(inputs.numpy(), self.flush)
        k = fz_arr(self.kernel.numpy(), self.flush)

        outputs = tiled_matmul(i, k, flush=self.flush)

        if self.use_bias:
            outputs = fz_arr(outputs + fz_arr(self.bias.numpy(), self.flush), self.flush)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs
