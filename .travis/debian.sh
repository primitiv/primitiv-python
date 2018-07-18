#!/bin/bash
set -xe

# before_install
docker pull debian:stable
docker run --name travis-ci -v $TRAVIS_BUILD_DIR:/primitiv-python -td debian:stable /bin/bash

# install
docker exec travis-ci bash -c "apt update"
docker exec travis-ci bash -c "apt install -y build-essential cmake python3-dev python3-pip python3-numpy"
docker exec travis-ci bash -c "pip3 install cython scikit-build"

# TODO(vbkaisetsu):
# Debian stretch contains Eigen 3.3.2. It has a bug around EIGEN_MPL2_ONLY
# mode and SparseCholesky module. It is fixed in newer version.
#
# For more details, see: http://eigen.tuxfamily.org/bz/show_bug.cgi?id=1392

# install Eigen
docker exec travis-ci bash -c "apt install -y wget"
docker exec travis-ci bash -c "wget http://bitbucket.org/eigen/eigen/get/3.3.4.tar.bz2 -O ./eigen.tar.bz2"
docker exec travis-ci bash -c "mkdir ./eigen"
docker exec travis-ci bash -c "tar xf ./eigen.tar.bz2 -C ./eigen --strip-components 1"
docker exec travis-ci bash -c "mkdir ./eigen/build"
docker exec travis-ci bash -c "cd ./eigen/build && cmake .."
docker exec travis-ci bash -c "cd ./eigen/build && make && make install"

# install OpenCL environment
docker exec travis-ci bash -c "apt install -y opencl-headers git pkg-config libhwloc-dev libltdl-dev ocl-icd-dev ocl-icd-opencl-dev clang-3.8 llvm-3.8-dev libclang-3.8-dev libz-dev"
docker exec travis-ci bash -c "wget https://github.com/CNugteren/CLBlast/archive/1.2.0.tar.gz -O ./clblast.tar.gz"
docker exec travis-ci bash -c "mkdir ./clblast"
docker exec travis-ci bash -c "tar xf ./clblast.tar.gz -C ./clblast --strip-components 1"
docker exec travis-ci bash -c "cd ./clblast && cmake . && make && make install"
# pocl 0.13 does not contain mem_fence() function that is used by primitiv.
# We build the latest pocl instead of using distribution's package.
# See: https://github.com/pocl/pocl/issues/294
docker exec travis-ci bash -c "git clone https://github.com/pocl/pocl.git"
docker exec travis-ci bash -c "cd ./pocl && cmake . -DCMAKE_INSTALL_PREFIX=/usr"
docker exec travis-ci bash -c "cd ./pocl && make && make install"

if [ "${WITH_CORE_LIBRARY}" = "yes" ]; then
    # script
    docker exec travis-ci bash -c "cd /primitiv-python && ./setup.py build --enable-eigen --enable-opencl -- -DCMAKE_VERBOSE_MAKEFILE=ON"
    docker exec travis-ci bash -c "cd /primitiv-python && ./setup.py build_ext -i --enable-eigen --enable-opencl -- -DCMAKE_VERBOSE_MAKEFILE=ON"
    docker exec travis-ci bash -c "cd /primitiv-python && ./setup.py test --enable-eigen --enable-opencl"

    # test installing by "pip install"
    docker exec travis-ci bash -c "cd /primitiv-python && ./setup.py sdist --bundle-core-library"

    docker exec travis-ci bash -c "pip3 install /primitiv-python/dist/primitiv-*.tar.gz --verbose --global-option --enable-eigen --global-option --enable-opencl"
    docker exec travis-ci bash -c "python3 -c 'import primitiv; dev = primitiv.devices.Naive(); dev = primitiv.devices.Eigen()'"
    docker exec travis-ci bash -c "pip3 uninstall -y primitiv"

    docker exec travis-ci bash -c "pip3 install --user /primitiv-python/dist/primitiv-*.tar.gz --verbose --global-option --enable-eigen --global-option --enable-opencl"
    docker exec travis-ci bash -c "python3 -c 'import primitiv; dev = primitiv.devices.Naive(); dev = primitiv.devices.Eigen()'"
    docker exec travis-ci bash -c "pip3 uninstall -y primitiv"

    # test installing by "./setup.py install"
    docker exec travis-ci bash -c "cd /primitiv-python && ./setup.py install --enable-eigen --enable-opencl"
    docker exec travis-ci bash -c "python3 -c 'import primitiv; dev = primitiv.devices.Naive(); dev = primitiv.devices.Eigen()'"
    docker exec travis-ci bash -c "pip3 uninstall -y primitiv"
else
    # install core library
    docker exec travis-ci bash -c "cd /primitiv-python/primitiv-core && cmake . -DPRIMITIV_USE_EIGEN=ON -DPRIMITIV_USE_OPENCL=ON -DCMAKE_VERBOSE_MAKEFILE=ON"
    docker exec travis-ci bash -c "cd /primitiv-python/primitiv-core && make"
    docker exec travis-ci bash -c "cd /primitiv-python/primitiv-core && make install"
    docker exec travis-ci bash -c "ldconfig"

    # script
    docker exec travis-ci bash -c "cd /primitiv-python && ./setup.py build --enable-eigen --enable-opencl --no-build-core-library"
    docker exec travis-ci bash -c "cd /primitiv-python && ./setup.py build_ext -i --enable-eigen --enable-opencl --no-build-core-library"
    docker exec travis-ci bash -c "cd /primitiv-python && ./setup.py test --enable-eigen --enable-opencl --no-build-core-library"

    # test installing by "./setup.py install"
    docker exec travis-ci bash -c "cd /primitiv-python && ./setup.py install --enable-eigen --enable-opencl --no-build-core-library"
    docker exec travis-ci bash -c "python3 -c 'import primitiv; dev = primitiv.devices.Naive(); dev = primitiv.devices.Eigen()'"
    docker exec travis-ci bash -c "pip3 uninstall -y primitiv"
fi

# after_script
docker stop travis-ci
