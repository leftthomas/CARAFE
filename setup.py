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
            ['carafe/carafe_cuda.cpp', 'carafe/carafe_cuda_kernel.cu'],
            extra_compile_args={
                'cxx': [],
                'nvcc': NVCC_ARGS
            }),
        CUDAExtension(
            'carafe_naive_cuda',
            ['carafe/carafe_naive_cuda.cpp', 'carafe/carafe_naive_cuda_kernel.cu'],
            extra_compile_args={
                'cxx': [],
                'nvcc': NVCC_ARGS
            }),
        CppExtension(
            'carafe_ext',
            ['carafe/carafe_ext.cpp']),
        CppExtension(
            'carafe_naive_ext',
            ['carafe/carafe_naive_ext.cpp'])
    ],
    cmdclass={'build_ext': BuildExtension},
    packages=[''],
    zip_safe=True,
    install_requires=['torch'])
