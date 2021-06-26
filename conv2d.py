import tvm
import numpy as np
from tvm import te
from tvm.topi.nn.conv2d import conv2d_nchw

# ref: https://tvm.apache.org/docs/tutorials/language/schedule_primitives.html
#      https://tvm.apache.org/docs/vta/tutorials/optimize/convolution_opt.html

def schedule_opt(conv):
    s = te.create_schedule(conv.op)
    # data = s[conv].op.input_tensors[0]
    # s[data].compute_inline()

    n, f, y, x = s[conv].op.axis
    # rc, ry, rx = s[conv].op.reduce_axis
    z = s[conv].fuse(x, y)
    s[conv].parallel(z)

    return s

def schedule_default(conv):
    return te.create_schedule(conv.op)

# ic表示input channel，oc表示output channel
def test_topi_conv2d(schedule, name):
    # 声明输入输出的大小
    n, ic, ih, iw = 8, 32, 64, 64
    oc, kh, kw = 32, 3, 3
    # n, ic, ih, iw = 1, 3, 32, 32
    # oc, kh, kw = 32, 3, 3
    # n, ic, ih, iw = 100, 512, 32, 32
    # oc, kh, kw = 1024, 3, 3
    dtype = 'float32'
    # 声明卷积的一些参数
    oh = ih + 2 - kh + 1
    ow = iw + 2 - kw + 1
    # 声明占位符
    A = te.placeholder(shape=(n, ic, ih, iw), dtype=dtype, name='A')
    B = te.placeholder(shape=(oc, ic, kh, kw), dtype=dtype, name='B')

    # 调用conv2d_nchw来进行conv2d的计算。
    conv = conv2d_nchw(A, B, (1, 1), (1, 1), (1, 1))

    # 这里需要大家调用tvm有的原语进行loop循环的优化，大家自己去补充
    s = schedule(conv)

    # 编译生成可执行的模块
    func_cpu = tvm.build(s, [A, B, conv], target="llvm")

    # 这个打印进行schedule优化后中间的ir
    with open(f'{name}.ir', 'w') as ir_file:
        ir_file.write(str(tvm.lower(s, [A, B, conv], simple_mode=True)))
    
    # 生成数据
    a_np = np.random.uniform(-1, 1, size=(n, ic, ih, iw)).astype(dtype)
    b_np = np.random.uniform(-1, 1, size=(oc, ic, kh, kw)).astype(dtype)

    # 指定底层的运行的硬件
    ctx = tvm.context("llvm", 0)
    d_cpu = tvm.nd.array(np.zeros((n, oc, oh, ow), dtype=dtype), ctx)

    # 进行转换
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(b_np, ctx)
    # 执行代码
    func_cpu(a, b, d_cpu)
    # 测试时间
    evaluator = func_cpu.time_evaluator(func_cpu.entry_name, ctx, number=5)
    # 打印时间
    eval_time = evaluator(a, b, d_cpu).mean * 1000.0
    print(f'{name} Conv: {eval_time} ms')
    return eval_time

def main():
    default_time = test_topi_conv2d(schedule_default, 'default')
    opt_time = test_topi_conv2d(schedule_opt, 'opt')
    print((default_time - opt_time) / default_time)

if __name__ == '__main__':
    main()



   


