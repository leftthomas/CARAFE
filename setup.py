from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

NVCC_ARGS = [
    '-D__CUDA_NO_HALF_OPERATORS__',
    '-D__CUDA_NO_HALF_CONVERSIONS__',
    '-D__CUDA_NO_HALF2_OPERATORS__',
]

VERSION = '0.0.4'

setup(
    name='carafe',
    version=VERSION,
    license='MIT',
    ext_modules=[
        CUDAExtension(
            'carafe_cuda',
            ['src/carafe_cuda.cpp', 'src/carafe_cuda_kernel.cu'],
            extra_compile_args={
                'cxx': [],
                'nvcc': NVCC_ARGS
            }),
        CUDAExtension(
            'carafe_naive_cuda',
            ['src/carafe_naive_cuda.cpp', 'src/carafe_naive_cuda_kernel.cu'],
            extra_compile_args={
                'cxx': [],
                'nvcc': NVCC_ARGS
            }),
        CppExtension(
            'carafe_ext',
            ['src/carafe_ext.cpp']),
        CppExtension(
            'carafe_naive_ext',
            ['src/carafe_naive_ext.cpp'])
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=['torch'])
