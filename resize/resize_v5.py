import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import cv2
import cv2
def resize_cpu(image, dsize):
    return cv2.resize(image, dsize=dsize)

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
    
    //printf("%d, %d, %f\\n", iHeight, oHeight, float(iHeight) / float(oHeight));
    const int dx = ((float)x * scale.x);
    const int dy = ((float)y * scale.y);
    //printf("%f, %f\\n", scale.x, scale.y);
    int count = 0;
    float sum = 0;

    
    for (size_t i = dx + scale.x/2; i < dx + scale.x/2; ++i) {
       for (size_t j = dy + scale.y/2; j < dy + scale.y/2; ++j) {
          if (i >= iWidth || j >= iHeight) { 
                 continue;
          }
          const uint8_t px = input[j * iWidth + i];
          sum += px;
          count += 1;
       }
    }
    //printf("%f, %d, %d, %d \\n", sum, count, (int)floor(sum / count + 0.5f), y*oWidth+x);
    output[y*oWidth+x] =  (uint8_t)floor(sum / count + 0.5f);
}
""")

input = cv2.imread("./lena.png", 0)
print(input.shape)
print(input[0])
#exit(0)
input = resize_cpu(input, (1280, 760))
#input = np.ones((512, 512), dtype=np.uint8)
target_size = 512
ref = resize_cpu(input, (target_size, target_size))
output = np.zeros((target_size, target_size), dtype=np.uint8) # specify output type to uint8
print(input.shape)
print(output.shape)
gpuResize = mod.get_function("gpuResize")

gpuResize(drv.In(input), np.int32(input.shape[1]), np.int32(input.shape[0]), drv.Out(output), np.int32(output.shape[1]), np.int32(output.shape[0]),block=(8, 8, 1), grid=(256, 256, 1))
print(input[0:16, 0:16].sum() / 16 / 16)
print(input[0:16, 16:32].sum() / 16 / 16)
print(input[7:9, 7:9].mean())
print(output[0])
print(ref[0])
print((output == ref).all())
