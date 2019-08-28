from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os;
import sys;
os.environ['PYTHONPATH'] = './install/Lib/site-packages'
setup(
    name='chamfer',
    ext_modules=[
        CUDAExtension('chamfer', [
            'chamfer_cuda.cpp',
            'chamfer.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
