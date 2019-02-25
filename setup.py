#!/usr/bin/env python3

import os
import sys

from distutils.dir_util import copy_tree
from setuptools.extension import Extension

import numpy as np

from Cython.Build import build_ext

VERSION = "0.4.0"

SUBMODULE_DIR = "primitiv-core"
EIGEN_DIR = "eigen-headers"

SUBMODULE_CMAKELIST = os.path.join(SUBMODULE_DIR, "CMakeLists.txt")
EIGEN_HEADER_DIR = os.path.join(EIGEN_DIR, "Eigen")

build_number = os.getenv("PRIMITIV_PYTHON_BUILD_NUMBER")
if build_number is not None:
    version_full = VERSION + "." + build_number
else:
    version_full = VERSION

bundle_eigen_headers = False
if "--bundle-eigen-headers" in sys.argv:
    i = sys.argv.index("--bundle-eigen-headers")
    sys.argv.pop(i)
    eigen_path = sys.argv.pop(i)
    copy_tree(eigen_path, EIGEN_DIR)
    bundle_eigen_headers = True

dirname = os.path.dirname(os.path.abspath(__file__))

if "--no-build-core-library" in sys.argv:
    build_core = False
    sys.argv.remove("--no-build-core-library")
else:
    build_core = os.path.exists(os.path.join(dirname, SUBMODULE_CMAKELIST))

eigen_bundled_exists = os.path.exists(os.path.join(dirname, EIGEN_HEADER_DIR))
if "--disable-eigen" in sys.argv:
    enable_eigen = False
    sys.argv.remove("--disable-eigen")
else:
    enable_eigen = eigen_bundled_exists

if build_core:
    import skbuild
    from skbuild import setup
else:
    from setuptools import setup

bundle_core_library = False
if "--bundle-core-library" in sys.argv:
    if not build_core:
        print("%s is not found" % SUBMODULE_CMAKELIST, file=sys.stderr)
        print("", file=sys.stderr)
        print("Run the following command to download primitiv core library:",
              file=sys.stderr)
        print("  git submodule update --init", file=sys.stderr)
        print("", file=sys.stderr)
        sys.exit(1)
    bundle_core_library = True
    sys.argv.remove("--bundle-core-library")

enable_cuda = False
if "--enable-cuda" in sys.argv:
    enable_cuda = True
    sys.argv.remove("--enable-cuda")

if "--enable-eigen" in sys.argv:
    enable_eigen = True
    sys.argv.remove("--enable-eigen")

enable_opencl = False
if "--enable-opencl" in sys.argv:
    enable_opencl = True
    sys.argv.remove("--enable-opencl")


def ext_common_args(*args, libraries=[], **kwargs):
    if build_core:
        libs = ["primitiv"]
        libs.extend(libraries)
        return Extension(
            *args, **kwargs,
            language="c++",
            libraries=libs,
            library_dirs=[os.path.join(skbuild.constants.CMAKE_INSTALL_DIR, "lib")],
            include_dirs=[
                np.get_include(),
                os.path.join(skbuild.constants.CMAKE_INSTALL_DIR, "include"),
                os.path.join(dirname, "primitiv"),
            ],
            extra_compile_args=["-std=c++11"],
        )
    else:
        return Extension(
            *args, **kwargs,
            language="c++",
            libraries=["primitiv"],
            include_dirs=[
                np.get_include(),
                os.path.join(dirname, "primitiv"),
            ],
            extra_compile_args=["-std=c++11"],
        )


ext_modules = [
    ext_common_args("primitiv._shape",
                    sources=["primitiv/_shape.pyx"]),
    ext_common_args("primitiv._tensor",
                    sources=["primitiv/_tensor.pyx"]),
    ext_common_args("primitiv._device",
                    sources=["primitiv/_device.pyx"]),
    ext_common_args("primitiv.devices._naive_device",
                    sources=["primitiv/devices/_naive_device.pyx"]),
    ext_common_args("primitiv._parameter",
                    sources=["primitiv/_parameter.pyx"]),
    ext_common_args("primitiv._initializer",
                    sources=["primitiv/_initializer.pyx"]),
    ext_common_args("primitiv.initializers._initializer_impl",
                    sources=["primitiv/initializers/_initializer_impl.pyx"]),
    ext_common_args("primitiv._graph",
                    sources=["primitiv/_graph.pyx"]),
    ext_common_args("primitiv._optimizer",
                    sources=["primitiv/_optimizer.pyx"]),
    ext_common_args("primitiv.optimizers._optimizer_impl",
                    sources=["primitiv/optimizers/_optimizer_impl.pyx"]),
    ext_common_args("primitiv._function",
                    sources=["primitiv/_function.pyx"]),
    ext_common_args("primitiv._model",
                    sources=["primitiv/_model.pyx"]),
    ext_common_args("primitiv.config",
                    sources=["primitiv/config.pyx"]),
]

if enable_cuda:
    ext_modules.append(
        ext_common_args(
            "primitiv.devices._cuda_device",
            libraries=[
                "cublas",
                "cudart",
                "curand",
                "pthread",
                "rt",
            ],
            sources=["primitiv/devices/_cuda_device.pyx"],
        )
    )

if enable_eigen:
    ext_modules.append(
        ext_common_args(
            "primitiv.devices._eigen_device",
            sources=["primitiv/devices/_eigen_device.pyx"],
        )
    )

if enable_opencl:
    ext_modules.append(
        ext_common_args(
            "primitiv.devices._opencl_device",
            libraries=[
                "clblast",
                "OpenCL",
            ],
            sources=["primitiv/devices/_opencl_device.pyx"],
        )
    )

setup_kwargs = {}
if build_core:
    setup_kwargs["cmake_source_dir"] = SUBMODULE_DIR
    setup_kwargs["cmake_install_dir"] = "./"
    setup_kwargs["setup_requires"] = [
        "cmake>=0.9.0",
        "cython>=0.29.5",
        "scikit-build>=0.6.1",
    ]
    setup_kwargs["cmake_args"] = ["-DPRIMITIV_BUILD_STATIC_LIBRARY=ON"]
    if sys.platform == "darwin":
        # NOTE(vbkaisetsu):
        # scikit-build adds -DCMAKE_OSX_DEPLOYMENT_TARGET with the default target if it does not
        # set manually. However scikit-build does not check cmake_args argument of setup()
        # for the target.
        try:
            cmake_args_pos = sys.argv.index("--")
        except ValueError:
            cmake_args_pos = len(sys.argv)
            sys.argv.append("--")
        sys.argv.insert(cmake_args_pos + 1, "-DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=10.12")
    if enable_cuda:
        setup_kwargs["cmake_args"].append("-DPRIMITIV_USE_CUDA=ON")
    if enable_eigen:
        setup_kwargs["cmake_args"].append("-DPRIMITIV_USE_EIGEN=ON")
        if eigen_bundled_exists:
            setup_kwargs["cmake_args"].append("-DEIGEN3_INCLUDE_DIR=%s" % os.path.join(dirname, EIGEN_DIR))
    if enable_opencl:
        setup_kwargs["cmake_args"].append("-DPRIMITIV_USE_OPENCL=ON")

with open(os.path.join(dirname, "MANIFEST.in"), "w") as fp:
    print("include README.md package_description.rst primitiv/py_optimizer.h", file=fp)
    print("recursive-include primitiv *.pyx *.pxd", file=fp)
    if bundle_core_library:
        print("recursive-include %s *" % SUBMODULE_DIR, file=fp)
    if bundle_eigen_headers:
        print("include %s/COPYING.* %s/README.md" % (EIGEN_DIR, EIGEN_DIR), file=fp)
        print("recursive-include %s *" % EIGEN_HEADER_DIR, file=fp)

setup(
    name="primitiv",
    version=version_full,
    description="primitiv: A Neural Network Toolkit. (Python frontend)",
    long_description=open(os.path.join(dirname,
                                       "package_description.rst")).read(),
    url="https://github.com/primitiv/primitiv-python",
    author="primitiv developer group",
    author_email="primitiv-developer-group@googlegroups.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    packages=[
        "primitiv",
        "primitiv.devices",
        "primitiv.initializers",
        "primitiv.optimizers",
    ],
    install_requires=[
        "numpy>=1.16.1",
    ],
    **setup_kwargs,
)
