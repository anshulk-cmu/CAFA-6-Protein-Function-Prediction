"""
setup.py
Build configuration for Smith-Waterman CUDA extension
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# CUDA compiler flags
extra_compile_args = {
    'cxx': ['-O3'],
    'nvcc': [
        '-O3',
        '--use_fast_math',
        '-arch=sm_80',  # Ampere architecture (RTX A6000)
        '-lineinfo'
    ]
}

setup(
    name='smith_waterman_cuda',
    ext_modules=[
        CUDAExtension(
            name='smith_waterman_cuda',
            sources=[
                'smith_waterman_kernel.cu',
                'smith_waterman_wrapper.cpp',
            ],
            extra_compile_args=extra_compile_args
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.20.0',
        'tqdm>=4.60.0'
    ],
    python_requires='>=3.9',
    description='GPU-accelerated Smith-Waterman sequence alignment',
    author='Anshul Kumar',
    author_email='anshulk@andrew.cmu.edu',
)
