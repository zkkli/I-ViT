import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn import Parameter

from .quant_utils import *


class QuantLinear(nn.Linear):
    """
    Class to quantize weights of given Linear layer

    Parameters:
    ----------
    weight_bit : int
        Bitwidth for quantized weights.
    bias_bit : int, default None
        Bitwidth for quantized bias.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    quant_mode : 'none' or 'symmetric', default 'none'
        The mode for quantization. 'none' for no quantization.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        weight_bit=8,
        bias_bit=32,
        per_channel=True,
        quant_mode="symmetric",
    ):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.weight_bit = weight_bit
        self.per_channel = per_channel
        self.bias_bit = bias_bit
        self.quantize_bias = False if bias_bit is None else True
        self.quant_mode = quant_mode

        if self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

        self.register_buffer("fc_scaling_factor", torch.zeros(self.out_features))
        self.register_buffer("weight_integer", torch.zeros_like(self.weight))
        if self.bias is not None:
            self.register_buffer("bias_integer", torch.zeros_like(self.bias))

    def __repr__(self):
        s = super(QuantLinear, self).__repr__()
        s = (
            "("
            + s
            + " weight_bit={}, quant_mode={})".format(self.weight_bit, self.quant_mode)
        )
        return s

    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, x, prev_act_scaling_factor=None):
        with torch.no_grad():
            w = self.weight
            if self.per_channel:
                v = w.reshape(w.shape[0], -1)
                cur_min = v.min(axis=1).values
                cur_max = v.max(axis=1).values
                self.min_val = cur_min
                self.max_val = cur_max
            else:
                raise Exception("For weight, we only support per_channel quantization.")

            self.fc_scaling_factor = symmetric_linear_quantization_params(
                self.weight_bit, self.min_val, self.max_val
            )

        self.weight_integer = self.weight_function(
            self.weight, self.weight_bit, self.fc_scaling_factor, True
        )

        prev_act_scaling_factor = prev_act_scaling_factor.to(
            x.device
        )  # @LeeJiho 24.09.09 : maching device
        bias_scaling_factor = self.fc_scaling_factor * prev_act_scaling_factor

        if self.bias is not None:
            self.bias_integer = self.weight_function(
                self.bias, self.bias_bit, bias_scaling_factor, True
            )
        else:
            self.bias_integer = None

        prev_act_scaling_factor = prev_act_scaling_factor.view(1, -1)
        x_int = x / prev_act_scaling_factor

        return (
            F.linear(x_int, weight=self.weight_integer, bias=self.bias_integer)
            * bias_scaling_factor,
            bias_scaling_factor,
        )


class QuantAct(nn.Module):
    """
    Class to quantize given activations
    Parameters:
    ----------
    activation_bit : int
        Bitwidth for quantized activations.
    act_range_momentum : float, default 0.95
        Momentum for updating the activation quantization range.
    running_stat : bool, default True
        Whether to use running statistics for activation quantization range.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    channel_len : int, default None
        Specify the channel length when using the per_channel mode.
    quant_mode : 'none' or 'asymmetric', default 'none'
        The mode for quantization. 'none' for no quantization.
    """

    def __init__(
        self,
        activation_bit=8,
        act_range_momentum=0.95,
        running_stat=True,
        per_channel=False,
        quant_mode="symmetric",
    ):
        super(QuantAct, self).__init__()

        self.activation_bit = activation_bit
        self.act_range_momentum = act_range_momentum
        self.running_stat = running_stat
        self.quant_mode = quant_mode
        self.per_channel = per_channel

        self.min_val = torch.zeros(1)
        self.max_val = torch.zeros(1)
        self.register_buffer("act_scaling_factor", torch.zeros(1))

        self.quant_mode = quant_mode
        self.per_channel = per_channel

        if self.quant_mode == "symmetric":
            self.act_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError(
                "unsupported quant mode: {}".format(self.quant_mode)
            )
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

    def __repr__(self):
        return f"{self.__class__.__name__}(activation_bit={self.activation_bit}, quant_mode: {self.quant_mode}, Act_min: {self.min_val.item()}, Act_max: {self.max_val.item()})"

    def fix(self):
        """
        fix the activation range by setting running stat
        """
        self.running_stat = False

    def unfix(self):
        """
        unfix the activation range by setting running stat
        """
        self.running_stat = True

    def forward(
        self,
        x,
        pre_act_scaling_factor=None,
        identity=None,
        identity_scaling_factor=None,
    ):
        # collect runnng stats
        with torch.no_grad():
            x_act = x if identity is None else identity + x
            if self.running_stat:
                if len(x_act.shape) == 4:
                    x_act = x_act.permute(0, 2, 3, 1)
                v = x_act.reshape(-1, x_act.shape[-1])
                v = v.transpose(0, 1)

                cur_min = v.min(axis=1).values
                cur_max = v.max(axis=1).values
                if torch.eq(self.min_val, self.max_val).all():
                    self.min_val = cur_min
                    self.max_val = cur_max
                else:
                    self.min_val = self.min_val * self.act_range_momentum + cur_min * (
                        1 - self.act_range_momentum
                    )
                    self.max_val = self.max_val * self.act_range_momentum + cur_max * (
                        1 - self.act_range_momentum
                    )
                self.max_val = self.max_val.max()
                self.min_val = self.min_val.min()

            self.act_scaling_factor = symmetric_linear_quantization_params(
                self.activation_bit, self.min_val, self.max_val
            )

        if pre_act_scaling_factor is None:
            # this is for the input quantization
            quant_act_int = self.act_function(
                x, self.activation_bit, self.act_scaling_factor, False
            )
        else:
            quant_act_int = fixedpoint_mul.apply(
                x,
                pre_act_scaling_factor,
                self.activation_bit,
                self.quant_mode,
                self.act_scaling_factor,
                identity,
                identity_scaling_factor,
            )

        correct_output_scale = self.act_scaling_factor.view(-1).to(x.device)
        # @LeeJiho 24.09.09 : maching device

        return quant_act_int * correct_output_scale, self.act_scaling_factor


class QuantMatMul(nn.Module):
    """
    Class to quantize weights of given matmul layer
    """

    def __init__(self):
        super(QuantMatMul, self).__init__()
        self.register_buffer("act_scaling_factor", torch.zeros(1))

    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, A, pre_act_scaling_factor_A, B, pre_act_scaling_factor_B):
        pre_act_scaling_factor_A = pre_act_scaling_factor_A.to(A.device)
        pre_act_scaling_factor_B = pre_act_scaling_factor_B.to(B.device)
        # @LeeJiho 24.09.09 : maching device

        A_int = A / pre_act_scaling_factor_A
        B_int = B / pre_act_scaling_factor_B
        act_scaling_factor = pre_act_scaling_factor_A * pre_act_scaling_factor_B
        self.act_scaling_factor = act_scaling_factor
        return (A_int @ B_int) * act_scaling_factor, act_scaling_factor


class QuantConv2d(nn.Conv2d):
    """
    Class to quantize weights of given convolutional layer
    Parameters:
    ----------
    weight_bit : int, default 4
        Bitwidth for quantized weights.
    bias_bit : int, default None
        Bitwidth for quantized bias.
    full_precision_flag : bool, default False
        If True, use fp32 and skip quantization
    quant_mode : 'symmetric' or 'asymmetric', default 'symmetric'
        The mode for quantization.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    fix_flag : bool, default False
        Whether the module is in fixed mode or not.
    weight_percentile : float, default 0
        The percentile to setup quantization range, 0 means no use of percentile, 99.9 means to cut off 0.1%.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        weight_bit=8,
        bias_bit=32,
        quant_mode="symmetric",
        per_channel=True,
        weight_percentile=0,
    ):
        super(QuantConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.weight_percentile = weight_percentile
        self.bias_bit = bias_bit
        self.quantize_bias = False if bias_bit is None else True

        self.register_buffer("conv_scaling_factor", torch.zeros(self.out_channels))
        self.register_buffer("weight_integer", torch.zeros_like(self.weight))
        self.register_buffer("bias_integer", torch.zeros_like(self.bias))

    def __repr__(self):
        s = super(QuantConv2d, self).__repr__()
        s = (
            "("
            + s
            + " weight_bit={}, quant_mode={})".format(self.weight_bit, self.quant_mode)
        )
        return s

    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, x, pre_act_scaling_factor=None):
        if self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError(
                "unsupported quant mode: {}".format(self.quant_mode)
            )
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

        with torch.no_grad():
            w = self.weight
            if self.per_channel:
                v = w.reshape(w.shape[0], -1)
                cur_min = v.min(axis=1).values
                cur_max = v.max(axis=1).values
                self.min_val = cur_min
                self.max_val = cur_max
            else:
                raise Exception("For weight, we only support per_channel quantization.")

            self.conv_scaling_factor = symmetric_linear_quantization_params(
                self.weight_bit, self.min_val, self.max_val
            )

        self.weight_integer = self.weight_function(
            self.weight, self.weight_bit, self.conv_scaling_factor, True
        )
        pre_act_scaling_factor = pre_act_scaling_factor.to(x.device)
        # @LeeJiho 24.09.09 : maching device
        bias_scaling_factor = self.conv_scaling_factor * pre_act_scaling_factor
        self.bias_integer = self.weight_function(
            self.bias, self.bias_bit, bias_scaling_factor, True
        )

        pre_act_scaling_factor = pre_act_scaling_factor.view(1, -1, 1, 1)
        x_int = x / pre_act_scaling_factor
        correct_output_scale = bias_scaling_factor.view(1, -1, 1, 1)

        return (
            F.conv2d(
                x_int,
                self.weight_integer,
                self.bias_integer,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            * correct_output_scale,
            correct_output_scale,
        )


class IntLayerNorm(nn.LayerNorm):
    """
    Implementation of I-LayerNorm
    Class to quantize given LayerNorm layer
    """

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(IntLayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)
        self.dim_sqrt = None
        self.register_buffer("norm_scaling_factor", torch.zeros(1))
        self.register_buffer("bias_integer", torch.zeros_like(self.bias))

    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, x, scaling_factor=None):
        if self.dim_sqrt is None:
            n = torch.tensor(x.shape[2], dtype=torch.float)
            self.dim_sqrt = torch.sqrt(n).cuda()

        # Normalization: computes mean and variance(std)
        scaling_factor = scaling_factor.to(
            x.device
        )  # @LeeJiho 24.09.09 : maching device
        x_int = x / scaling_factor
        mean_int = round_ste.apply(x_int.mean(axis=2, keepdim=True))
        y_int = x_int - mean_int
        y_sq_int = y_int**2
        var_int = torch.sum(y_sq_int, axis=2, keepdim=True)

        # Integer Iteration
        k = 2**16
        for _ in range(10):
            k_1 = floor_ste.apply((k + floor_ste.apply(var_int / k)) / 2)
            k = k_1
        std_int = k

        factor = floor_ste.apply((2**31 - 1) / std_int)
        y_int = floor_ste.apply(y_int * factor / 2)
        scaling_factor = self.dim_sqrt / 2**30

        # scaling and shifting
        bias = self.bias.data.detach() / (self.weight.data.detach())
        bias_int = floor_ste.apply(bias / scaling_factor)

        self.bias_integer = bias_int

        y_int = y_int + bias_int
        scaling_factor = scaling_factor * self.weight
        x = y_int * scaling_factor
        self.norm_scaling_factor = scaling_factor
        return x, scaling_factor


class IntGELU(nn.Module):
    """
    Implementation of ShiftGELU
    Class to quantize given GELU layer
    """

    def __init__(self, output_bit=8, intgelu_exp_n=23):
        super(IntGELU, self).__init__()
        self.output_bit = output_bit

        self.n = intgelu_exp_n  # sufficiently large integer
        # The minimum value for ensuring accuracy (varies depending on models)
        print("IntGELU    | n: ", self.n)

        self.register_buffer("act_scaling_factor", torch.zeros(1))

    def fix(self):
        pass

    def unfix(self):
        pass

    def int_exp_shift(self, x_int, scaling_factor):
        x_int = x_int + floor_ste.apply(x_int / 2) - floor_ste.apply(x_int / 2**4)

        with torch.no_grad():
            x0_int = torch.floor(-1.0 / scaling_factor)
        x_int = torch.max(x_int, self.n * x0_int)

        q = floor_ste.apply(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int = r / 2 - x0_int
        exp_int = torch.clamp(floor_ste.apply(exp_int * 2 ** (self.n - q)), min=0)
        scaling_factor = scaling_factor / 2**self.n

        return exp_int, scaling_factor

    def forward(self, x, scaling_factor=None):
        scaling_factor = scaling_factor.to(
            x.device
        )  # @LeeJiho 24.09.09 : maching device
        pre_x_int = x / scaling_factor
        scaling_factor_sig = scaling_factor * 1.702

        x_int_max, _ = pre_x_int.max(dim=-1, keepdim=True)
        x_int = pre_x_int - x_int_max

        exp_int, _ = self.int_exp_shift(x_int, scaling_factor_sig)  # e^(x-x_max)

        exp_int_max, _ = self.int_exp_shift(
            -x_int_max, scaling_factor_sig
        )  # e^(-x_max)
        exp_int_sum = exp_int + exp_int_max

        exp_int_sum.clamp_max_(2**31 - 1)
        factor = floor_ste.apply((2**31 - 1) / exp_int_sum)
        sigmoid_int = floor_ste.apply(
            exp_int * factor / 2 ** (31 - self.output_bit + 1)
        )
        sigmoid_scaling_factor = torch.Tensor([1 / 2 ** (self.output_bit - 1)]).cuda()

        x_int = pre_x_int * sigmoid_int
        scaling_factor = scaling_factor * sigmoid_scaling_factor
        self.act_scaling_factor = scaling_factor
        return x_int * scaling_factor, scaling_factor


class IntSoftmax(nn.Module):
    """
    Implementation of Shiftmax
    Class to quantize given Softmax layer
    """

    def __init__(self, output_bit=8, intsoftmax_exp_n=15):
        super(IntSoftmax, self).__init__()
        self.output_bit = output_bit

        self.n = intsoftmax_exp_n  # sufficiently large integer
        # The minimum value for ensuring accuracy (varies depending on models)
        print("IntSoftmax | n: ", self.n)
        self.register_buffer("act_scaling_factor", torch.zeros(1))

    def fix(self):
        pass

    def unfix(self):
        pass

    def int_exp_shift(self, x_int, scaling_factor):
        x_int = x_int + floor_ste.apply(x_int / 2) - floor_ste.apply(x_int / 2**4)

        with torch.no_grad():
            x0_int = torch.floor(-1.0 / scaling_factor)
        x_int = torch.max(x_int, self.n * x0_int)

        q = floor_ste.apply(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int = r / 2 - x0_int
        exp_int = torch.clamp(floor_ste.apply(exp_int * 2 ** (self.n - q)), min=0)
        scaling_factor = scaling_factor / 2**self.n
        return exp_int, scaling_factor

    def forward(self, x, scaling_factor):
        scaling_factor = scaling_factor.to(
            x.device
        )  # @LeeJiho 24.09.09 : maching device
        x_int = x / scaling_factor
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max

        exp_int, _ = self.int_exp_shift(x_int, scaling_factor)
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)

        exp_int_sum.clamp_max_(2**31 - 1)
        factor = floor_ste.apply((2**31 - 1) / exp_int_sum)
        exp_int = floor_ste.apply(exp_int * factor / 2 ** (31 - self.output_bit + 1))
        scaling_factor = torch.Tensor([1 / 2 ** (self.output_bit - 1)]).cuda()

        self.act_scaling_factor = scaling_factor
        return exp_int * scaling_factor, scaling_factor


class Log2_Int_Quantizer(nn.Module):
    def __init__(self):
        super().__init__()
        """ log sqrt 2 quantizer for attention map """

        self.activation_bit = 4

        self.n_levels = 2**self.activation_bit
        self.int_bias = torch.tensor(1.0)

    def __repr__(self):
        return f"{self.__class__.__name__}(activation_bit={self.activation_bit}"

    def int_log_quant_10x(self, x):
        x = x.to(torch.int32)
        zero_mask = x == 0
        log2_int = torch.full_like(x, -1, dtype=torch.int32)

        temp_x = x.clone()
        for i in range(15, -1, -1):
            shift = 1 << i
            greater_equal = temp_x >= shift
            log2_int += greater_equal.to(torch.int32)
            temp_x = temp_x >> greater_equal.to(torch.int32)

        fractional_add = torch.zeros_like(x, dtype=torch.int32)

        temp_x = x - (1 << log2_int)
        temp_x = temp_x << 1  # temp_x *= 2
        fractional_add += (temp_x >= (1 << log2_int)).to(torch.int32) * 5
        out = -1 * (log2_int * 10 + fractional_add)
        out[zero_mask] = 99999
        return out

    def int_log_dequant_10x(self, y):
        zero_mask = y == 99999
        y = -y

        int_part = y // 10
        frac_part = y % 10 / 5

        int_num = 1 << int_part
        frac_num = frac_part * (1 << (int_part - 1))
        out = (int_num + frac_num).floor()
        out[zero_mask] = 0
        return out

    def forward(self, x_hat: torch.Tensor, s_x: torch.Tensor):
        assert 0 <= x_hat.min() and x_hat.max() <= 1, f"{x_hat.min()} {x_hat.max()}"

        x_int = round_ste.apply(x_hat / s_x)

        # [1] log quantization in huge domain
        x_int_log_q = (self.int_log_quant_10x(x_int) // 10) * 10
        x_int_log_dq = self.int_log_dequant_10x(x_int_log_q)

        # ver2.  * Prec@1 69.454 Prec@5 89.464
        out = x_int_log_dq * 256 // x_int_log_dq.max()
        if out.unique().numel() > self.n_levels:
            out = out // 2
        out = out.clamp(0, 255)

        # print(out.unique().numel(), out.unique())
        # 10 tensor([  0.,   1.,   2.,   4.,   8.,  16.,  32.,  64., 128., 255.],

        assert (
            out.unique().numel() <= self.n_levels
        ), f"{out.unique().numel(), out.unique()}"

        s_x = s_x * 255

        x_hat = out * s_x

        return x_hat, s_x


class Log2_half_Int_Quantizer(nn.Module):
    def __init__(self):
        super().__init__()
        """ log sqrt 2 quantizer for attention map """

        self.activation_bit = 4

        self.n_levels = 2**self.activation_bit
        self.int_bias = torch.tensor(1.0)

    def __repr__(self):
        return f"{self.__class__.__name__}(activation_bit={self.activation_bit}"

    def int_log_quant_10x(self, x):
        x = x.to(torch.int32)
        zero_mask = x == 0
        log2_int = torch.full_like(x, -1, dtype=torch.int32)

        temp_x = x.clone()
        for i in range(15, -1, -1):
            shift = 1 << i
            greater_equal = temp_x >= shift
            log2_int += greater_equal.to(torch.int32)
            temp_x = temp_x >> greater_equal.to(torch.int32)

        fractional_add = torch.zeros_like(x, dtype=torch.int32)

        temp_x = x - (1 << log2_int)
        temp_x = temp_x << 1  # temp_x *= 2
        fractional_add += (temp_x >= (1 << log2_int)).to(torch.int32) * 5
        out = -1 * (log2_int * 10 + fractional_add)
        out[zero_mask] = 99999
        return out

    def int_log_dequant_10x(self, y):
        zero_mask = y == 99999
        y = -y

        int_part = y // 10
        frac_part = y % 10 / 5

        int_num = 1 << int_part
        frac_num = frac_part * (1 << (int_part - 1))
        out = (int_num + frac_num).floor()
        out[zero_mask] = 0
        return out

    def forward(self, x_hat: torch.Tensor, s_x: torch.Tensor):
        assert 0 <= x_hat.min() and x_hat.max() <= 1, f"{x_hat.min()} {x_hat.max()}"

        x_int = round_ste.apply(x_hat / s_x)

        # [1] log quantization in huge domain
        x_int_log_q = self.int_log_quant_10x(x_int)
        # print(x_int_log_q.unique().numel(), x_int_log_q.unique())
        # 32 tensor([ -155,  -150,  -145,  -140,  -135,  -130,  -125,  -120,  -115,  -110,
        #          -105,  -100,   -95,   -90,   -85,   -80,   -75,   -70,   -65,   -60,
        #           -55,   -50,   -45,   -40,   -35,   -30,   -25,   -20,   -15,   -10,
        #             0, 99999]

        # [2] log dequantization
        x_int_log_dq = self.int_log_dequant_10x(x_int_log_q)

        # print(x_int_log_dq.unique().numel(), x_int_log_dq.unique())
        # 32 tensor([0.0000e+00, 1.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00, 6.0000e+00,
        #         8.0000e+00, 1.2000e+01, 1.6000e+01, 2.4000e+01, 3.2000e+01, 4.8000e+01,
        #         6.4000e+01, 9.6000e+01, 1.2800e+02, 1.9200e+02, 2.5600e+02, 3.8400e+02,
        #         5.1200e+02, 7.6800e+02, 1.0240e+03, 1.5360e+03, 2.0480e+03, 3.0720e+03,
        #         4.0960e+03, 6.1440e+03, 8.1920e+03, 1.2288e+04, 1.6384e+04, 2.4576e+04,
        #         3.2768e+04, 4.9152e+04], device='cuda:0')

        # [3] [0, 255]
        # #  * Prec@1 69.146 Prec@5 89.320
        # x_int_log_dq = x_int_log_dq * x_int_log_dq.unique()[-16] // x_int_log_dq.max()
        # out = x_int_log_dq.clamp(0, 255)

        # ver2.  * Prec@1 69.454 Prec@5 89.464
        out = x_int_log_dq * 256 // x_int_log_dq.max()
        if out.unique().numel() > self.n_levels:
            print(out.unique().numel(), out.unique())
            out = out // 2
        out = out.clamp(0, 255)

        assert (
            out.unique().numel() <= self.n_levels
        ), f"{out.unique().numel(), out.unique()}"

        s_x = s_x * 255

        x_hat = out * s_x

        return x_hat, s_x


class Log2Quantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel

    for [0, 1)
    """

    def __init__(self, n_bits: int = 4, channel_wise: bool = False):
        super(Log2Quantizer, self).__init__()
        assert 2 <= n_bits <= 8, "bitwidth not supported"
        self.n_bits = n_bits
        self.n_levels = 2**self.n_bits
        self.delta = None
        self.inited = False
        self.channel_wise = channel_wise

    def forward(self, x: torch.Tensor, s_x):

        if self.inited is False:
            self.delta = self.init_quantization_scale(x)
            self.inited = True

        # start quantization
        x_dequant = self.quantize(x, self.delta)
        return x_dequant, s_x

    def init_quantization_scale(self, x: torch.Tensor):
        def lp_loss(pred, tgt, p=2.0, reduction="none"):
            """
            loss function measured in L_p Norm
            """
            if reduction == "none":
                return (pred - tgt).abs().pow(p).sum(1).mean()
            else:
                return (pred - tgt).abs().pow(p).mean()

        delta = None
        x_clone = x.clone().detach()
        delta = x_clone.max()
        best_score = 1e10
        for pct in [0.999, 0.9999, 0.99999]:  #
            try:
                new_delta = torch.quantile(x_clone.reshape(-1), pct)
            except:
                new_delta = torch.tensor(
                    np.percentile(x_clone.reshape(-1).cpu(), pct * 100),
                    device=x_clone.device,
                    dtype=torch.float32,
                )
            x_q = self.quantize(x_clone, new_delta)
            score = lp_loss(x_clone, x_q, p=2, reduction="all")

            if score < best_score:
                best_score = score
                delta = new_delta

        return delta

    def quantize(self, x, delta):
        from math import sqrt

        x_int = torch.round(-1 * (x / delta).log2())
        mask = x_int >= self.n_levels
        # print(x_int.unique())
        #   tensor([-1., -0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.,
        #   13., 14., 15., inf], device='cuda:0')
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        # print(x_quant.unique())
        #   tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
        #   14., 15.], device='cuda:0')
        x_float_q = 2 ** (-1 * x_quant) * delta
        x_float_q[mask] = 0
        # Log2 UINT8 test
        # print((x_float_q * 255).round().unique())
        # exit()
        return x_float_q


class LogSqrt2Quantizer(nn.Module):
    """
    From RepQ-ViT

    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel

    for [0, 1)
    """

    def __init__(self, n_bits: int = 4, channel_wise: bool = False):
        super(LogSqrt2Quantizer, self).__init__()
        assert 2 <= n_bits <= 8, "bitwidth not supported"
        self.n_bits = n_bits
        self.n_levels = 2**self.n_bits
        self.delta = None
        self.inited = False
        self.channel_wise = channel_wise

    def forward(self, x: torch.Tensor, s_x):

        if self.inited is False:
            self.delta = self.init_quantization_scale(x)
            self.inited = True

        # start quantization
        x_dequant = self.quantize(x, self.delta)
        return x_dequant, s_x

    def init_quantization_scale(self, x: torch.Tensor):
        def lp_loss(pred, tgt, p=2.0, reduction="none"):
            """
            loss function measured in L_p Norm
            """
            if reduction == "none":
                return (pred - tgt).abs().pow(p).sum(1).mean()
            else:
                return (pred - tgt).abs().pow(p).mean()

        delta = None
        x_clone = x.clone().detach()
        delta = x_clone.max()
        best_score = 1e10
        for pct in [0.999, 0.9999, 0.99999]:  #
            try:
                new_delta = torch.quantile(x_clone.reshape(-1), pct)
            except:
                new_delta = torch.tensor(
                    np.percentile(x_clone.reshape(-1).cpu(), pct * 100),
                    device=x_clone.device,
                    dtype=torch.float32,
                )
            x_q = self.quantize(x_clone, new_delta)
            score = lp_loss(x_clone, x_q, p=2, reduction="all")

            if score < best_score:
                best_score = score
                delta = new_delta

        return delta

    def quantize(self, x, delta):
        from math import sqrt

        x_int = torch.round(-1 * (x / delta).log2() * 2)
        mask = x_int >= self.n_levels
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        odd_mask = (x_quant % 2) * (sqrt(2) - 1) + 1
        x_float_q = 2 ** (-1 * torch.ceil(x_quant / 2)) * odd_mask * delta
        x_float_q[mask] = 0

        return x_float_q
