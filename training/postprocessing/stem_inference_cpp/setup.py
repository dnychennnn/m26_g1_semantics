from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

setup(name='stem_inference_cpp',
      ext_modules=[CppExtension(name='stem_inference_cpu',
                                 sources=['cast_votes_cpu.cpp', 'common.cpp']),
                   CUDAExtension(name='stem_inference_cuda',
                                sources=['cast_votes_cuda.cu', 'common.cpp'])],
      cmdclass={'build_ext': BuildExtension})


# setup(name='stem_inference_cpp',
      # ext_modules=[CUDAExtension(name='stem_inference_cuda',
                                # sources=['cast_votes_cuda.cu', 'common.cpp'])],
      # cmdclass={'build_ext': BuildExtension})

