"""Simple Layer DSL wrapper to ease creation of neural nets."""
from dataclasses import dataclass
from tvm import relay
from collections import namedtuple

import numpy as np
from tvm.relay.op.tensor import exp


QConfig = namedtuple('QConfig', 'from_dtype, from_scale, from_zero_point, \
                                input_dtype, input_scale, input_zero_point, \
                                kernel_dtype, kernel_scale, kernel_zero_point, \
                                output_dtype, output_scale, output_zero_point',
                      defaults=('int32', 65.0, 0.0, 'int8', 8.0, 0.0, 'int8', 8.0, 0.0, 'int32', 74.0, 0.0))

class QuantizeContext(object):
    qconfig_dict = dict()
    qconfig_print = dict()
    default_qconfig = QConfig()

    @staticmethod
    def read_qconfig_from_file(file_path):
        pass

    @staticmethod
    def set_default_qconfig(qconfig):
        QuantizeContext.default_qconfig = qconfig

def get_qconfig(name):
    #print(QuantizeContext.qconfig_dict)
    if name in QuantizeContext.qconfig_dict:
        return QuantizeContext.qconfig_dict[name]
    else:
        QuantizeContext.qconfig_print[name] = QuantizeContext.default_qconfig
        return QuantizeContext.default_qconfig


def quantized_conv2d(data,
                     kernel_dtype,
                     name,
                     input_channels,
                     kernel_size,
                     output_channels,
                     strides=(1, 1),
                     padding=(0, 0),
                     weight=None,
                     add_bias=False,
                     input_scale=8.0,
                     kernel_scale=8.0,
                     input_zero_point=0.0,
                     kernel_zero_point=0.0,
                     data_layout='NCHW',
                     kernel_layout='OIHW',
                     **kwargs):

    """Wrapper of qnn.conv2d
    Parameters
    ----------
    data : relay.Expr
        The input expression.
    weight : relay.Expr
        The weight to conv2d.
    name : str
        The name of this convolution.
    input_channels: int
        The number of input channels.
    out_channels: int
        The number of output channels.
    input_scale : float
        The scale of input.
    kernel_scale : float
        The scale of kernel.
    input_zero_point : float
        The zero point of input.
    kernel_zero_point : float
        The zero point of kernel.
    kwargs : dict
        Additional arguments.
    Returns
    -------
    result : relay.Expr
        The result.
    """

    # print("%s, %s, %d, %d, %d, %d, %d" % (, kernel_dtype, input_channels, output_channels, kernel_size[0], strides[0], padding[0]))

    input_zero_point = relay.const(input_zero_point, 'int32')
    kernel_zero_point = relay.const(kernel_zero_point, 'int32')

    if isinstance(input_scale, float):
        input_scale = relay.const(input_scale, 'float32')
    else:
        input_scale = relay.const(input_scale.astype('float32'), 'float32')

    if isinstance(kernel_scale, float):
        kernel_scale = relay.const(kernel_scale, 'float32')
    else:
        kernel_scale = relay.const(kernel_scale.astype('float32'), 'float32')

    if kernel_layout == "OIHW":
        kernel_shape = (output_channels, input_channels, kernel_size[0], kernel_size[1])
    elif kernel_layout == "HWIO":
        kernel_shape = (kernel_size[0], kernel_size[1], input_channels, output_channels)
    elif kernel_layout == "HWOI":
        kernel_shape = (kernel_size[0], kernel_size[1], output_channels, input_channels)
    elif kernel_layout == "OHWI":
        kernel_shape = (output_channels, kernel_size[0], kernel_size[1], input_channels)
    else:
        raise RuntimeError("Unsupported kernel layout {}".format(kernel_layout))

    if weight is None:
        weight = relay.var(name + "_weight", shape=kernel_shape, dtype=kernel_dtype)

    conv2d = relay.qnn.op.conv2d(data, weight, input_zero_point, kernel_zero_point, input_scale, kernel_scale,
                                kernel_size=kernel_size, channels=output_channels, data_layout=data_layout, kernel_layout=kernel_layout, strides=strides, padding=padding, **kwargs)

    if add_bias:
        if data_layout == 'NCHW':
            bias_shape = (1, output_channels, 1, 1)
        elif data_layout == 'NHWC':
            bias_shape = (1, 1, 1, output_channels)
        elif data_layout == 'HWCN':
            bias_shape = (1, 1, output_channels, 1)
        elif data_layout == 'HWNC':
            bias_shape = (1, 1, 1, output_channels)
        else:
            raise RuntimeError("Unsupported conv2d layout {}".format(data_layout))

        bias = relay.var(name + "_bias", shape=bias_shape, dtype="int32")
        return relay.add(conv2d, bias)
    else:
        return conv2d


def quantize(data,
             output_scale=8.0,
             output_zero_point=0.0,
             axis=-1,
             out_dtype='int8'):

    output_scale = relay.const(output_scale, 'float32')
    output_zero_point = relay.const(output_zero_point, 'int32')

    return relay.qnn.op.quantize(data, output_scale, output_zero_point, axis, out_dtype)


def requantize(data,
               input_scale=8.0,
               input_zero_point=0.0,
               output_scale=8.0,
               output_zero_point=0.0,
               axis=-1,
               rounding="None",
               compute_dtype="None",
               out_dtype="int8"):

    if isinstance(input_scale, float):
        input_scale = relay.const(input_scale, 'float32')
    else:
        input_scale = relay.const(np.array(input_scale).astype('float32'))

    input_zero_point = relay.const(input_zero_point, 'int32')

    if isinstance(output_scale, float):
        output_scale = relay.const(output_scale, 'float32')
    else:
        output_scale = relay.const(np.array(output_scale).astype('float32'))

    output_zero_point = relay.const(output_zero_point, 'int32')

    return relay.qnn.op.requantize(data,
                                   input_scale,
                                   input_zero_point,
                                   output_scale,
                                   output_zero_point,
                                   axis,
                                   rounding,
                                   compute_dtype,
                                   out_dtype)


def dequantize(data,
               input_scale,
               input_zero_point=0.0,
               axis=-1):

    if isinstance(input_scale, float):
        input_scale = relay.const(input_scale, 'float32')
    else:
        input_scale = relay.const(input_scale.astype('float32'), 'float32')

    input_zero_point = relay.const(input_zero_point, 'int32')

    return relay.qnn.op.dequantize(data,
               input_scale,
               input_zero_point,
               axis)


def add(lhs,
        rhs,
        lhs_scale,
        rhs_scale,
        output_scale,
        lhs_zero_point=0.0,
        rhs_zero_point=0.0,
        output_zero_point=0.0):

    lhs_scale = relay.const(lhs_scale, 'float32')
    lhs_zero_point = relay.const(lhs_zero_point, 'int32')

    rhs_scale = relay.const(rhs_scale, 'float32')
    rhs_zero_point = relay.const(rhs_zero_point, 'int32')

    if np.ndim(output_scale) == 1:
        output_scale = output_scale[0]
    if np.ndim(output_zero_point) == 1:
        output_zero_point = output_zero_point[0]

    output_scale = relay.const(output_scale, 'float32')
    output_zero_point = relay.const(output_zero_point, 'int32')

    return relay.qnn.op.add(lhs,
        rhs,
        lhs_scale,
        lhs_zero_point,
        rhs_scale,
        rhs_zero_point,
        output_scale,
        output_zero_point)


def quantized_dense(data,
          name,
          units,
          kernel_shape,
          kernel_dtype,
          input_scale=8.0,
          kernel_scale=8.0,
          input_zero_point=0.0,
          kernel_zero_point=0.0,
          add_bias=False,
          out_dtype="int32"):
    """Qnn Dense operator.
    Applies a quantized linear transformation
     .. math::
     `Y = X * W`
    Parameters
    ----------
    data : tvm.relay.Expr
        The quantized input data to the operator.
    weight : tvm.relay.Expr
        The quantized weight expressions.
    input_zero_point: tvm.relay.Expr
        The input zero point.
    kernel_zero_point: tvm.relay.Expr
        The kernel zero point.
    input_scale: tvm.relay.Expr
        The scale for the input tensor.
    kernel_scale: tvm.relay.Expr
        The scale for the weight tensor. The scale for the weight tensor is
        stored for access to this during relay. This information is not
        needed in the pass pipeline after qnn.conv2d is lowered to the
        sequence of steps as in nn.conv2d. See also input_scale in Requantize.
    units : int
        Number of hidden units of the dense transformation.
    out_dtype : str, optional
        Specifies the output data type for mixed precision dense can be int32 or int16.
    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    input_zero_point = relay.const(input_zero_point, 'int32')
    kernel_zero_point = relay.const(kernel_zero_point, 'int32')
    if isinstance(input_scale, float):
        input_scale = relay.const(input_scale, 'float32')
    else:
        input_scale = relay.const(input_scale.astype('float32'), 'float32')

    if isinstance(kernel_scale, float):
        kernel_scale = relay.const(kernel_scale, 'float32')
    else:
        kernel_scale = relay.const(kernel_scale.astype('float32'), 'float32')

    weight = relay.var(name + "_weight", shape=kernel_shape, dtype=kernel_dtype)

    dense = relay.qnn.op.dense(data,
                               weight,
                               input_zero_point,
                               kernel_zero_point,
                               input_scale,
                               kernel_scale,
                               units,
                               out_dtype)
    if add_bias:
        bias = relay.var(name + "_bias", dtype="int32")
        return relay.nn.bias_add(dense, bias, axis=-1)
    else:
        return dense


def quantized_matmul(x, y,
                     input_scale1,
                     input_scale2,
                     x_zero_point=0.0,
                     y_zero_point=0.0):
    x_zero_point = relay.const(x_zero_point, 'int32')
    y_zero_point = relay.const(y_zero_point, 'int32')     
    if isinstance(input_scale1, float):
        x_scale = relay.const(input_scale1, 'float32')
    else:
        x_scale = relay.const(input_scale1.astype('float32'), 'float32')
    if isinstance(input_scale2, float):
        y_scale = relay.const(input_scale2, 'float32')
    else:
        y_scale = relay.const(input_scale2.astype('float32'), 'float32')

    matmul = relay.qnn.op.batch_matmul(x, y, 
                                       x_zero_point, 
                                       y_zero_point, 
                                       x_scale, 
                                       y_scale, 
                                       out_dtype="int32")
    return matmul 


def quantized_layernorm(data, 
                        bias_int):
    mean = relay.mean(data, axis=2, keepdims=True)
    data = data - mean

    data = relay.cast(data, 'int32')
    data_sq = data * data

    data_sq = relay.cast(data_sq, 'uint32')
    var = relay.sum(data_sq, axis=2, keepdims=True)

    std = relay.const(2 ** 16, 'uint32')
    for _ in range(10):
        tmp = (std + var/std)/relay.const(2, 'uint32')
        std = tmp
    std = relay.cast(std, 'int32')

    factor = relay.const(2**31-1, 'int32')
    data =  (factor / std) * data / relay.const(2, 'int32')
    data = data + bias_int

    return data


def shift_exp(data, input_scale, n):

    data = data + relay.right_shift(data, relay.const(1, dtype='int32')) - relay.right_shift(data, relay.const(4, dtype='int32'))

    x0 = relay.const(-1.0/input_scale-1, 'int32')
    n = relay.const(n, dtype='int32')

    data = relay.maximum(data, n*x0)

    q = data / x0
    r = data - q * x0

    exp_int = relay.right_shift(r, relay.const(1, dtype='int32')) - x0
    exp_int = relay.left_shift(exp_int, n-q)

    return exp_int



def quantized_softmax(data, input_scale):
    data = relay.cast(data, 'int32')
    data_max = relay.max(data, axis=-1, keepdims=True)
    data = data - data_max

    exp_int = shift_exp(data, input_scale, 16)

    exp_int_sum = relay.sum(exp_int, axis=-1, keepdims=True)
    factor = relay.const(2**31-1, 'int32')
    # exp_int = (factor/exp_int_sum) * exp_int / relay.const(2 ** 24, 'int32')
    exp_int = relay.right_shift((factor/exp_int_sum) * exp_int, relay.const(24, dtype='int32'))

    exp_int = relay.cast(exp_int, 'int8')

    return exp_int


def quantized_gelu(pre_data, input_scale):
    pre_data = relay.cast(pre_data, 'int32')
    data_max = relay.max(pre_data, axis=-1, keepdims=True)
    data = pre_data - data_max

    exp_int = shift_exp(data, input_scale* 1.702, 23)
    exp_int_max = shift_exp(-data_max, input_scale* 1.702, 23)
    exp_int_sum = exp_int + exp_int_max

    factor = relay.const(2**31-1, 'int32')
    # sigmoid_int = (factor/exp_int_sum) * exp_int / relay.const(2 ** 24, 'int32')
    sigmoid_int = relay.right_shift((factor/exp_int_sum) * exp_int, relay.const(24, dtype='int32'))

    gelu = pre_data * sigmoid_int

    return gelu