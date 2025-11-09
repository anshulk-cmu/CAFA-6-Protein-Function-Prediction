"""
setup.py
Build Configuration for Smith-Waterman CUDA Extension
Phase 2A: CAFA6 Project - Compilation Instructions

Build Instructions:
    python setup.py install

Development Build (faster, no installation):
    python setup.py build_ext --inplace

Clean Build:
    python setup.py clean --all
    python setup.py install
"""

from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os
import sys

# ============================================================================
# Configuration
# ============================================================================

# Extension name (will be imported as: import smith_waterman_cuda)
EXTENSION_NAME = 'smith_waterman_cuda'

# Source files
SOURCES = [
    'smith_waterman_kernel.cu',      # CUDA kernel implementation
    'smith_waterman_wrapper.cpp',    # C++ PyTorch bindings
]

# Include directories
INCLUDE_DIRS = [
    os.path.dirname(os.path.abspath(__file__)),
]

# ============================================================================
# Compiler Flags
# ============================================================================

# C++ compiler flags
CXX_FLAGS = [
    '-O3',                    # Maximum optimization
    '-std=c++17',             # C++17 standard (PyTorch compatibility)
    '-fPIC',                  # Position-independent code (required for shared libraries)
]

# CUDA compiler (nvcc) flags
NVCC_FLAGS = [
    '-O3',                    # Maximum optimization
    '--use_fast_math',        # Fast math operations (trade precision for speed)
    '-lineinfo',              # Include line info for profiling with NSight Compute
    '--ptxas-options=-v',     # Verbose PTX assembly (shows register usage)

    # GPU Architecture: NVIDIA RTX A6000 (Ampere, compute capability 8.6)
    # Generate code for both PTX (forward compatibility) and SASS (performance)
    '-gencode=arch=compute_80,code=sm_80',    # Ampere baseline (A100)
    '-gencode=arch=compute_86,code=sm_86',    # Ampere (RTX A6000, RTX 3090)
    '-gencode=arch=compute_86,code=compute_86',  # PTX for future compatibility

    # Memory optimizations
    '--maxrregcount=64',      # Limit registers per thread (improves occupancy)

    # Debugging (comment out for production builds)
    # '-G',                   # Device debug mode (disables optimizations!)
    # '-g',                   # Host debug symbols
]

# Additional linker flags
EXTRA_LINK_ARGS = []

# Platform-specific adjustments
if sys.platform == 'linux':
    CXX_FLAGS.append('-Wno-unused-function')
    CXX_FLAGS.append('-Wno-sign-compare')
elif sys.platform == 'darwin':  # macOS
    # macOS doesn't support CUDA, but include for completeness
    print("Warning: CUDA not supported on macOS")
elif sys.platform == 'win32':  # Windows
    CXX_FLAGS = ['/O2', '/std:c++17']  # MSVC flags

# ============================================================================
# Dependency Checks
# ============================================================================

def check_cuda_available():
    """Verify CUDA is available and compatible"""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please install CUDA toolkit and PyTorch with CUDA support.\n"
            "Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu118"
        )

    cuda_version = torch.version.cuda
    print(f"✓ CUDA version: {cuda_version}")

    # Check compute capability
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        print(f"✓ GPU compute capability: {capability[0]}.{capability[1]}")

        # Warn if not Ampere or newer
        if capability[0] < 8:
            print(f"⚠ Warning: GPU compute capability {capability[0]}.{capability[1]} < 8.0")
            print("  This code is optimized for Ampere (sm_80+) architecture.")
            print("  Performance may be suboptimal on older GPUs.")

def check_dependencies():
    """Check required packages are installed"""
    required = {
        'torch': '2.0.0',
        'numpy': '1.20.0',
    }

    missing = []

    for package, min_version in required.items():
        try:
            if package == 'torch':
                import torch
                version = torch.__version__.split('+')[0]  # Remove +cu118 suffix
            elif package == 'numpy':
                import numpy
                version = numpy.__version__

            print(f"✓ {package}: {version}")
        except ImportError:
            missing.append(f"{package}>={min_version}")

    if missing:
        raise RuntimeError(
            f"Missing dependencies: {', '.join(missing)}\n"
            f"Install with: pip install {' '.join(missing)}"
        )

# Run checks
print("Checking build environment...")
check_dependencies()
check_cuda_available()
print()

# ============================================================================
# Build Extension
# ============================================================================

ext_modules = [
    CUDAExtension(
        name=EXTENSION_NAME,
        sources=SOURCES,
        include_dirs=INCLUDE_DIRS,
        extra_compile_args={
            'cxx': CXX_FLAGS,
            'nvcc': NVCC_FLAGS,
        },
        extra_link_args=EXTRA_LINK_ARGS,
    )
]

# ============================================================================
# Package Setup
# ============================================================================

setup(
    name='smith_waterman_cuda',
    version='1.0.0',
    author='Anshul Kumar',
    author_email='anshulk@andrew.cmu.edu',
    description='GPU-Accelerated Smith-Waterman Sequence Alignment (Phase 2A)',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',

    # Extension modules
    ext_modules=ext_modules,

    # Build system
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    },

    # Python package
    py_modules=['smith_waterman'],  # Include smith_waterman.py

    # Dependencies
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.20.0',
        'tqdm>=4.60.0',
    ],

    # Python version requirement
    python_requires='>=3.9',

    # PyPI classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C++',
        'Programming Language :: CUDA',
    ],

    # Keywords for searchability
    keywords='bioinformatics gpu cuda sequence-alignment smith-waterman protein',

    # Project URLs
    url='https://github.com/anshulk-cmu/CAFA-6-Protein-Function-Prediction',
    project_urls={
        'Source': 'https://github.com/anshulk-cmu/CAFA-6-Protein-Function-Prediction',
    },
)

# ============================================================================
# Post-Build Instructions
# ============================================================================

print("\n" + "="*80)
print("Build Configuration Summary")
print("="*80)
print(f"Extension name: {EXTENSION_NAME}")
print(f"Source files: {', '.join(SOURCES)}")
print(f"CUDA flags: {' '.join(NVCC_FLAGS[:3])}...")
print(f"Target architectures: sm_80 (A100), sm_86 (RTX A6000)")
print("="*80)
print("\nTo use after installation:")
print("  >>> import smith_waterman_cuda")
print("  >>> from smith_waterman import align_sequences")
print("  >>> score = align_sequences('ARNDCQEGH', 'ARDCQEG')")
print("\nFor profiling with NSight Compute:")
print("  $ ncu --set full -o profile python your_script.py")
print("="*80 + "\n")
