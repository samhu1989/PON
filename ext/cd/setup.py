from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os;
import sys;
os.environ['PYTHONPATH'] = os.path.abspath('./install/Lib/site-packages')+':'+os.environ['PYTHONPATH'];
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
