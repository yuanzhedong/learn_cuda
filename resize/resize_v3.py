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
    int sum = 0;
    for (size_t i = dx; i < dx + scale.x; ++i) {
       for (size_t j = dy; j < dy + scale.y; ++j) {
          if (i >= iWidth || j >= iHeight) { 
                 continue;
          }
          const uint8_t px = input[j * iWidth + i];
          sum += px;
          count += 1;
       }
    }
    output[y*oWidth+x] =  (int)floor((float) sum / count + 0.5f);
}
""")

#a = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
input = np.array([[1,1,2,2],
               [1,1,2,2],
               [3,3,4,3],
               [3,3,4,2]], dtype=np.uint8)
output = np.zeros((2, 2), dtype=np.uint8)

input = np.array([[3, 106, 107, 40, 148, 112, 254, 151],
                [62, 173, 91, 93, 33, 111, 139, 25],
                [99, 137, 80, 231, 101, 204, 74, 219],
                [240, 173, 85, 14, 40, 230, 160, 152],
                [230, 200, 177, 149, 173, 239, 103, 74],
                [19, 50, 209, 82, 241, 103, 3, 87],
                [252, 191, 55, 154, 171, 107, 6, 123],
                [7, 101, 168, 85, 115, 103, 32, 11]],
                dtype=np.uint8)
output = np.zeros((input.shape[1]//2, input.shape[0]//2), dtype=np.uint8)
output = np.zeros((input.shape[1], input.shape[0]), dtype=np.uint8)

print(input.shape)
print(output.shape)
gpuResize = mod.get_function("gpuResize")

gpuResize(drv.In(input), np.int32(input.shape[1]), np.int32(input.shape[0]), drv.Out(output), np.int32(output.shape[1]), np.int32(output.shape[0]),block=(8, 8, 1), grid=(256, 256, 1))
print(input)
print(output)


ref = np.array([[ 86, 83, 101, 142],
                [162, 103, 144, 151],
                [125, 154, 189,  67],
                [138, 116, 124,  43]])
print(ref)
print(np.equal(output, ref))
# ref = np.array([[1, 2],
#                 [3, 4]])
# print(np.equal(ref, output))
# #cuda-gdb --args python -m pycuda.debug demo.py
