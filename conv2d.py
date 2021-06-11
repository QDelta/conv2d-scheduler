import tvm
import numpy as np
from tvm import autotvm, te
from numbers import Integral
from tvm.topi.nn.util import get_pad_tuple
from tvm.topi.nn.conv2d import conv2d_nchw
from tvm.contrib import util

# ref: https://tvm.apache.org/docs/tutorials/language/schedule_primitives.html
#      https://tvm.apache.org/docs/vta/tutorials/optimize/convolution_opt.html

# def schedule_opt(output):
#     s = te.create_schedule(output.op)
#     # just blocking
#     na, oca, ha, wa = s[output].op.axis
#     oc_out, oc_inn = s[output].split(oca, factor = 8)
#     h_out, h_inn = s[output].split(ha, factor = 8)
#     w_out, w_inn = s[output].split(wa, factor = 8)
#     s[output].reorder(na, oc_out, h_out, w_out, oc_inn, h_inn, w_inn)
#     return s

def compute_opt(input, filter, stride, padding, dilation):
    pass

def compute_default(input, filter, stride, padding, dilation):
    return conv2d_nchw(input, filter, stride, padding, dilation)
    # defined in tvm-src/python/build/lib/tvm/topi/nn/conv2d.py

#ic表示input channel，oc表示output channel      
def test_topi_conv2d(compute, name):
    # 声明输入输出的大小
    n, ic, ih, iw = 8, 32, 64, 64
    oc, kh, kw = 64, 3, 3
    # n, ic, ih, iw = 1, 3, 32, 32
    # oc, kh, kw = 32, 3, 3
    # n, ic, ih, iw = 100, 512, 32, 32
    # oc, kh, kw = 1024, 3, 3
    dtype = 'float32'
    # 声明卷积的一些参数
    stride_h, stride_w = 1, 1
    pad_h, pad_w = 1, 1
    dilation_h, dilation_w = 1, 1
    oh = (ih + 2 * pad_h - kh) // stride_h + 1
    ow = (iw + 2 * pad_w - kw) // stride_w + 1
    # 声明占位符
    A = te.placeholder(shape=(n, ic, ih, iw), dtype=dtype, name='A')
    B = te.placeholder(shape=(oc, ic, kh, kw), dtype=dtype, name='B')

    # 调用conv2d_nchw来进行conv2d的计算。
    # output = conv2d_nchw(Input = A, Filter = B, stride = (stride_h, stride_w), padding = (pad_h, pad_w), dilation = (dilation_h, dilation_w))
    output = compute(A, B, (stride_h, stride_w), (pad_h, pad_w), (dilation_h, dilation_w))

    # 这一句是调用tvm默认的schedule函数，表示不加任何优化的schedule
    s = te.create_schedule(output.op)

    # 这里需要大家调用tvm有的原语进行loop循环的优化，大家自己去补充
    # s = schedule(output)

    # 编译生成可执行的模块
    func_cpu = tvm.build(s, [A, B, output], target="llvm")

    # 这个打印进行schedule优化后中间的ir
    with open(f'{name}.ir', 'w') as ir_file:
        ir_file.write(str(tvm.lower(s, [A, B, output], simple_mode=True)))
    
    # 生成数据
    a_np = np.random.uniform(-1, 1, size=(n, ic, ih, iw)).astype(dtype)
    b_np = np.random.uniform(-1, 1, size=(oc, ic, kh, kw)).astype(dtype)

    # 指定底层的运行的硬件
    ctx = tvm.context("llvm",0) 
    d_cpu = tvm.nd.array(np.zeros((n, oc, oh, ow), dtype=dtype), ctx)

    # 进行转换
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(b_np, ctx)
    # 执行代码
    func_cpu(a, b, d_cpu)
    # 测试时间
    evaluator = func_cpu.time_evaluator(func_cpu.entry_name, ctx, number=2)
    # 打印时间
    eval_time = evaluator(a, b, d_cpu).mean * 1000.0
    print(f'{name} Conv: {eval_time} ms')
    return eval_time

def main():
    default_time = test_topi_conv2d(compute_default, 'default')
    opt_time = test_topi_conv2d(compute_opt, 'opt')
    print((default_time - opt_time) / default_time)

if __name__ == '__main__':
    main()



   


