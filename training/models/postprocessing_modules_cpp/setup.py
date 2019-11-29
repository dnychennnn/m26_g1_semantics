from torch.utils import cpp_extension
from setuptools import setup, Extension

setup(name='postprocessing_modules_cpp',
      ext_modules=[cpp_extension.CUDAExtension(name='postprocessing_modules_cpp',
              sources=['cast_votes.cpp', 'common.cpp'],
              extra_compile_args={'cxx': ['-g'],
                                  'nvcc': ['-02']})],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
