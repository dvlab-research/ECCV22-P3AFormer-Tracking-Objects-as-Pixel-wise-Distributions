## TransCenter: Transformers with Dense Queries for Multiple-Object Tracking
## Copyright Inria
## Year 2021
## Contact : yihong.xu@inria.fr
##
## TransCenter is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.

## TransCenter is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program, TransCenter.  If not, see <http://www.gnu.org/licenses/> and the LICENSE file.
##
##
## TransCenter has code derived from
## (1) 2020 fundamentalvision.(Apache License 2.0: https://github.com/fundamentalvision/Deformable-DETR)
## (2) 2020 Philipp Bergmann, Tim Meinhardt. (GNU General Public License v3.0 Licence: https://github.com/phil-bergmann/tracking_wo_bnw)
## (3) 2020 Facebook. (Apache License Version 2.0: https://github.com/facebookresearch/detr/)
## (4) 2020 Xingyi Zhou.(MIT License: https://github.com/xingyizhou/CenterTrack)
##
## TransCenter uses packages from
## (1) 2019 Charles Shang. (BSD 3-Clause Licence: https://github.com/CharlesShang/DCNv2)
## (2) 2017 NVIDIA CORPORATION. (Apache License, Version 2.0: https://github.com/NVIDIA/flownet2-pytorch/tree/master/networks/correlation_package)
## (3) 2019 Simon Niklaus. (GNU General Public License v3.0: https://github.com/sniklaus/pytorch-liteflownet)
## (4) 2018 Tak-Wai Hui. (Copyright (c), see details in the LICENSE file: https://github.com/twhui/LiteFlowNet)
import torch
from torch.nn.modules.module import Module
from torch.autograd import Function
import correlation_cuda

class CorrelationFunction(Function):

    # def __init__(self, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=2, corr_multiply=1):
    #     super(CorrelationFunction, self).__init__()
    #     self.pad_size = pad_size
    #     self.kernel_size = kernel_size
    #     self.max_displacement = max_displacement
    #     self.stride1 = stride1
    #     self.stride2 = stride2
    #     self.corr_multiply = corr_multiply
    #     # self.out_channel = ((max_displacement/stride2)*2 + 1) * ((max_displacement/stride2)*2 + 1)

    @staticmethod
    def forward(ctx, input1, input2, pad_size, kernel_size, max_displacement,stride1, stride2, corr_multiply):
        ctx.save_for_backward(input1, input2)
        ctx.pad_size = pad_size
        ctx.kernel_size = kernel_size
        ctx.max_displacement = max_displacement
        ctx.stride1 = stride1
        ctx.stride2 = stride2
        ctx.corr_multiply = corr_multiply

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            output = input1.new()

            correlation_cuda.forward(input1, input2, rbot1, rbot2, output,
                                     pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()

            grad_input1 = input1.new()
            grad_input2 = input2.new()

            correlation_cuda.backward(input1, input2, rbot1, rbot2, grad_output, grad_input1, grad_input2,
                                      ctx.pad_size, ctx.kernel_size, ctx.max_displacement, ctx.stride1, ctx.stride2, ctx.corr_multiply)

        return grad_input1, grad_input2, None, None, None, None, None, None


class Correlation(Module):
    def __init__(self, pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    # @staticmethod
    def forward(self, input1, input2):

        input1 = input1.contiguous()
        input2 = input2.contiguous()
        # result = CorrelationFunction(self.pad_size, self.kernel_size, self.max_displacement,self.stride1, self.stride2, self.corr_multiply)(input1, input2)
        result = CorrelationFunction.apply(input1, input2, self.pad_size, self.kernel_size, self.max_displacement,self.stride1, self.stride2, self.corr_multiply)

        return result

