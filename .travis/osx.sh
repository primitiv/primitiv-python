#!/bin/bash
set -xe

# install
brew update
brew upgrade python
brew install eigen
pip3 install cython numpy scikit-build

pushd $TRAVIS_BUILD_DIR

mkdir work

if [ "${WITH_CORE_LIBRARY}" = "yes" ]; then
    # script
    git submodule update --init
    ./setup.py build --enable-eigen -- -DCMAKE_VERBOSE_MAKEFILE=ON
    ./setup.py build_ext -i --enable-eigen -- -DCMAKE_VERBOSE_MAKEFILE=ON
    ./setup.py test --enable-eigen

    # test installing by "pip install"
    ./setup.py sdist --bundle-core-library

    pip3 install dist/primitiv-*.tar.gz --verbose --global-option --enable-eigen
    pushd work
    python3 -c 'import primitiv; dev = primitiv.devices.Naive(); dev = primitiv.devices.Eigen()'
    popd
    pip3 uninstall -y primitiv

    pip3 install --user dist/primitiv-*.tar.gz --verbose --global-option --enable-eigen
    pushd work
    python3 -c 'import primitiv; dev = primitiv.devices.Naive(); dev = primitiv.devices.Eigen()'
    popd
    pip3 uninstall -y primitiv

    # test installing by "./setup.py install"
    ./setup.py install --enable-eigen
    pushd work
    python3 -c 'import primitiv; dev = primitiv.devices.Naive(); dev = primitiv.devices.Eigen()'
    popd
    pip3 uninstall -y primitiv
else
    pushd primitiv-core
    cmake . -DPRIMITIV_USE_EIGEN=ON -DCMAKE_VERBOSE_MAKEFILE=ON
    make
    make install
    popd
    ./setup.py build --enable-eigen --no-build-core-library
    ./setup.py build_ext -i --enable-eigen --no-build-core-library
    ./setup.py test --enable-eigen --no-build-core-library

    # test installing by "./setup.py install"
    ./setup.py install --enable-eigen --no-build-core-library
    pushd work
    python3 -c 'import primitiv; dev = primitiv.devices.Naive(); dev = primitiv.devices.Eigen()'
    popd
    pip3 uninstall -y primitiv
fi

