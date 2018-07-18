#!/bin/bash
set -xe

# before_install
docker pull fedora:latest
docker run --name travis-ci -v $TRAVIS_BUILD_DIR:/primitiv-python -td fedora:latest /bin/bash

# install
docker exec travis-ci bash -c "dnf update -y"
docker exec travis-ci bash -c "dnf install -y rpm-build gcc-c++ cmake make python3-devel python3-numpy eigen3-devel"
docker exec travis-ci bash -c "pip3 install cython scikit-build"

# NOTE(vbkaisetsu):
# OpenCL test is disabled because irreproducible memory error is occured
# in Python+OpenCL+Fedora(+Travis?) combination.
#
# For developers:
# If you have Fedora machine and the bug is reproducible, please fix the bug
# in your environment.

# # install OpenCL environment
# docker exec travis-ci bash -c "dnf install -y opencl-headers hwloc-devel libtool-ltdl-devel ocl-icd-devel ocl-icd clang llvm-devel clang-devel zlib-devel blas-devel boost-devel patch --setopt=install_weak_deps=False"
# docker exec travis-ci bash -c "git clone https://github.com/clMathLibraries/clBLAS.git"
# docker exec travis-ci bash -c "cd ./clBLAS/src && cmake . -DCMAKE_INSTALL_PREFIX=/usr -DBUILD_TEST=OFF -DBUILD_KTEST=OFF"
# docker exec travis-ci bash -c "cd ./clBLAS/src && make && make install"
# # pocl 0.13 does not contain mem_fence() function that is used by primitiv.
# # We build the latest pocl instead of using distribution's package.
# # See: https://github.com/pocl/pocl/issues/294
# docker exec travis-ci bash -c "git clone https://github.com/pocl/pocl.git"
# docker exec travis-ci bash -c "cd ./pocl && cmake . -DCMAKE_INSTALL_PREFIX=/usr"
# docker exec travis-ci bash -c "cd ./pocl && make && make install"

if [ "${WITH_CORE_LIBRARY}" = "yes" ]; then
    # script
#     docker exec travis-ci bash -c "cd /primitiv-python && ./setup.py build --enable-eigen --enable-opencl"
#     docker exec travis-ci bash -c "cd /primitiv-python && ./setup.py test --enable-eigen --enable-opencl"
    docker exec travis-ci bash -c "cd /primitiv-python && ./setup.py build --enable-eigen -- -DCMAKE_VERBOSE_MAKEFILE=ON"
    docker exec travis-ci bash -c "cd /primitiv-python && ./setup.py build_ext -i --enable-eigen -- -DCMAKE_VERBOSE_MAKEFILE=ON"
    docker exec travis-ci bash -c "cd /primitiv-python && ./setup.py test --enable-eigen"

    # test installing by "pip install"
    docker exec travis-ci bash -c "cd /primitiv-python && ./setup.py sdist --bundle-core-library"

#     docker exec travis-ci bash -c "pip3 install --user /primitiv-python/dist/primitiv-*.tar.gz --verbose --global-option --enable-eigen --global-option --enable-opencl"
    docker exec travis-ci bash -c "pip3 install --user /primitiv-python/dist/primitiv-*.tar.gz --verbose --global-option --enable-eigen"
    docker exec travis-ci bash -c "python3 -c 'import primitiv; dev = primitiv.devices.Naive(); dev = primitiv.devices.Eigen()'"
    docker exec travis-ci bash -c "pip3 uninstall -y primitiv"

    # test installing by "./setup.py install"
#     docker exec travis-ci bash -c "cd /primitiv-python && ./setup.py install --enable-eigen --enable-opencl"
    docker exec travis-ci bash -c "cd /primitiv-python && ./setup.py install --enable-eigen"
    docker exec travis-ci bash -c "python3 -c 'import primitiv; dev = primitiv.devices.Naive(); dev = primitiv.devices.Eigen()'"
    docker exec travis-ci bash -c "pip3 uninstall -y primitiv"
else
    # install core library
#     docker exec travis-ci bash -c "cd /primitiv-python/primitiv-core && cmake . -DPRIMITIV_USE_EIGEN=ON -DPRIMITIV_USE_OPENCL=ON"
    docker exec travis-ci bash -c "cd /primitiv-python/primitiv-core && cmake . -DPRIMITIV_USE_EIGEN=ON -DCMAKE_VERBOSE_MAKEFILE=ON"
    docker exec travis-ci bash -c "cd /primitiv-python/primitiv-core && make"
    docker exec travis-ci bash -c "cd /primitiv-python/primitiv-core && make install"

    # script
#     docker exec travis-ci bash -c "cd /primitiv-python && ./setup.py build --enable-eigen --enable-opencl --no-build-core-library"
#     docker exec travis-ci bash -c "export LD_LIBRARY_PATH=/usr/local/lib && cd /primitiv-python && ./setup.py test --enable-eigen --enable-opencl --no-build-core-library"
    docker exec travis-ci bash -c "cd /primitiv-python && ./setup.py build --enable-eigen --no-build-core-library"
    docker exec travis-ci bash -c "cd /primitiv-python && ./setup.py build_ext -i --enable-eigen --no-build-core-library"
    docker exec travis-ci bash -c "export LD_LIBRARY_PATH=/usr/local/lib && cd /primitiv-python && ./setup.py test --enable-eigen --no-build-core-library"

    # test installing by "./setup.py install"
#     docker exec travis-ci bash -c "cd /primitiv-python && ./setup.py install --enable-eigen --enable-opencl --no-build-core-library"
    docker exec travis-ci bash -c "cd /primitiv-python && ./setup.py install --enable-eigen --no-build-core-library"
    docker exec travis-ci bash -c "export LD_LIBRARY_PATH=/usr/local/lib && python3 -c 'import primitiv; dev = primitiv.devices.Naive(); dev = primitiv.devices.Eigen()'"
    docker exec travis-ci bash -c "pip3 uninstall -y primitiv"
fi

# after_script
docker stop travis-ci
