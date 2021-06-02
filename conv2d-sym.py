import tvm
from tvm import te
from tvm.topi.nn.conv2d import conv2d_nchw

# just symbolic
def test_topi_conv2d():
    n, ic, ih, iw = te.var('n'), te.var('ic'), te.var('ih'), te.var('iw')
    oc, kh, kw = te.var('oc'), te.var('kh'), te.var('kw')
    
    dtype = 'float32'

    stride_h, stride_w = 1, 1
    pad_h, pad_w = 1, 1
    dilation_h, dilation_w = 1, 1

    A = te.placeholder(shape=(n, ic, ih, iw), dtype=dtype, name='A')
    B = te.placeholder(shape=(oc, ic, kh, kw), dtype=dtype, name='B')
    
    output = conv2d_nchw(Input = A, Filter = B, stride = (stride_h, stride_w), padding = (pad_h, pad_w), dilation = (dilation_h, dilation_w))

    s = te.create_schedule(output.op)

    with open('sym.ir', 'w') as ir_file:
        ir_file.write(str(tvm.lower(s, [A, B, output], simple_mode=True)))

    print(s[output].op.axis)

   
def main():
    test_topi_conv2d()
    
if __name__ == '__main__':
    main()



   


