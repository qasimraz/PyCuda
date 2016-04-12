import numpy as np
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray

kernel = SourceModule("""
__global__ void twice(float *x)
{
    const unsigned int i = threadIdx.x + threadIdx.y*blockDim.x;
    x[i]=2*x[i];
}
""")

twice = kernel.get_function('twice')
x = np.random.randn(16).astype(np.float32)
x_gpu = gpuarray.to_gpu(x)
twice(x_gpu, block=(4, 4, 1), grid = (1, 1))
print x, np.sum(x)
print x_gpu.get(), np.float32(gpuarray.sum(x_gpu).get())
