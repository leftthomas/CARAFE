from setuptools import setup, find_packages

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

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
            'carafe_ext',
            ['carafe/src/carafe_ext.cpp', 'carafe/src/carafe_cuda.cpp', 'carafe/src/carafe_cuda_kernel.cu'],
            extra_compile_args={
                'cxx': [],
                'nvcc': NVCC_ARGS
            }),
        CUDAExtension(
            'carafe_naive_ext',
            ['carafe/src/carafe_naive_ext.cpp', 'carafe/src/carafe_naive_cuda.cpp', 'carafe/src/carafe_naive_cuda_kernel.cu'],
            extra_compile_args={
                'cxx': [],
                'nvcc': NVCC_ARGS
            })
    ],
    packages=find_packages(exclude=('test',)),
    package_data={'.': ['*.so']},
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
    install_requires=['torch>=1.5'])
