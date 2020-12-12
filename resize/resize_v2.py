import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

mod = SourceModule \
    (
    """
    #include <stdint.h>
    
__global__ void gpuResize( uint8_t* input, int iWidth, int iHeight, uint8_t* output, int oWidth, int oHeight )
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const float2 scale = make_float2( float(iWidth) / float(oWidth), float(iHeight) / float(oHeight) );
    
    if( x >= oWidth || y >= oHeight )
        return;
    
    const int dx = ((float)x * scale.x);
    const int dy = ((float)y * scale.y);

    int count = 0;
    for (size_t i = dx; i < dx + scale.x; ++i) {
       for (size_t j = dy; j < dy + scale.y; ++j) {
          if (i >= iWidth || j >= iHeight) { 
                 continue;
          }
          const uint8_t px = input[j * iWidth + i];
          output[y*oWidth+x] += px;
          count += 1;
       }
    }
    output[y*oWidth+x] = output[y*oWidth+x] / count;
}
""")

#a = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
input = np.array([[1,1,2,2],
               [1,1,2,2],
               [3,3,4,3],
               [3,3,4,2]], dtype=np.uint8)

output = np.zeros((2, 2), dtype=np.uint8)
output.shape
gpuResize = mod.get_function("gpuResize")

gpuResize(drv.In(input), np.int32(4), np.int32(4), drv.Out(output), np.int32(2), np.int32(2),block=(8, 8, 1), grid=(256, 256, 1))
print(output)

ref = np.array([[1, 2],
                [3, 4]])
print(np.equal(ref, output))
#cuda-gdb --args python -m pycuda.debug demo.py
