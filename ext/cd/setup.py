from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os;
if not 'PYTHONPATH' in os.environ.keys():
    os.environ['PYTHONPATH'] = './install/Lib/site-packages';
else:
    os.environ['PYTHONPATH'] = './install/Lib/site-packages:' + os.environ['PYTHONPATH'];
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