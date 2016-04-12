import numpy
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

a_gpu = gpuarray.to_gpu(
numpy.random.randn(4,4).astype(numpy.float32))
a_doubled = (2∗a_gpu).get()
print a_doubled
print a_gpu