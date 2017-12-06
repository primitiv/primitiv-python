#!/bin/bash
set -xe

# before_install
docker pull debian:stable
docker run --name travis-ci -v $TRAVIS_BUILD_DIR:/primitiv-python -td debian:stable /bin/bash

# install
docker exec travis-ci bash -c "apt update"
docker exec travis-ci bash -c "apt install -y git build-essential cmake python3-dev python3-pip python3-numpy"
docker exec travis-ci bash -c "pip3 install cython scikit-build"

# install OpenCL environment
docker exec travis-ci bash -c "apt install -y opencl-headers libclblas-dev pkg-config libhwloc-dev libltdl-dev ocl-icd-dev ocl-icd-opencl-dev clang-3.8 llvm-3.8-dev libclang-3.8-dev libz-dev"
# pocl 0.13 does not contain mem_fence() function that is used by primitiv.
# We build the latest pocl instead of using distribution's package.
# See: https://github.com/pocl/pocl/issues/294
docker exec travis-ci bash -c "git clone https://github.com/pocl/pocl.git"
docker exec travis-ci bash -c "cd ./pocl && cmake . -DCMAKE_INSTALL_PREFIX=/usr"
docker exec travis-ci bash -c "cd ./pocl && make && make install"

if [ "${WITH_CORE_LIBRARY}" = "yes" ]; then
    # script
    docker exec travis-ci bash -c "cd /primitiv-python && git submodule update --init"
    docker exec travis-ci bash -c "cd /primitiv-python && ./setup.py build --enable-opencl && ./setup.py test"

    # test installing by "pip install"
    docker exec travis-ci bash -c "cd /primitiv-python && ./setup.py sdist --bundle-core-library"

    docker exec travis-ci bash -c "pip3 install /primitiv-python/dist/primitiv-*.tar.gz --verbose --global-option --enable-opencl"
    docker exec travis-ci bash -c "python3 -c 'import primitiv; dev = primitiv.devices.Naive()'"
    docker exec travis-ci bash -c "pip3 uninstall -y primitiv"

    docker exec travis-ci bash -c "pip3 install --user /primitiv-python/dist/primitiv-*.tar.gz --verbose --global-option --enable-opencl"
    docker exec travis-ci bash -c "python3 -c 'import primitiv; dev = primitiv.devices.Naive()'"
    docker exec travis-ci bash -c "pip3 uninstall -y primitiv"
else
    # install core library
    docker exec travis-ci bash -c "git clone https://github.com/primitiv/primitiv.git libprimitiv"
    docker exec travis-ci bash -c "cd ./libprimitiv && cmake . -DPRIMITIV_USE_OPENCL=ON"
    docker exec travis-ci bash -c "cd ./libprimitiv && make && make install"
    docker exec travis-ci bash -c "ldconfig"

    # script
    docker exec travis-ci bash -c "cd /primitiv-python && ./setup.py build --enable-opencl && ./setup.py test"
fi

# test installing by "./setup.py install"
docker exec travis-ci bash -c "cd /primitiv-python && ./setup.py install --enable-opencl"
docker exec travis-ci bash -c "python3 -c 'import primitiv; dev = primitiv.devices.Naive()'"
docker exec travis-ci bash -c "pip3 uninstall -y primitiv"

# after_script
docker stop travis-ci
