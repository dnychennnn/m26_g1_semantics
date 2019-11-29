from torch.utils import cpp_extension
from setuptools import setup, Extension

setup(name='stem_voting_cpp',
      ext_modules=[cpp_extension.CUDAExtension(name='stem_voting_cpp',
              sources=['stem_voting.cpp'],
              extra_compile_args={'cxx': ['-g'],
                                  'nvcc': ['-02']})],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
