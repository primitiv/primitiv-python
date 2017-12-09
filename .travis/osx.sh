#!/bin/bash
set -xe

# install
brew update
brew install python3
pip3 install cython numpy scikit-build

pushd $TRAVIS_BUILD_DIR

mkdir work

if [ "${WITH_CORE_LIBRARY}" = "yes" ]; then
    # script
    git submodule update --init
    ./setup.py build
    ./setup.py test

    # test installing by "pip install"
    ./setup.py sdist --bundle-core-library

    pip3 install dist/primitiv-*.tar.gz --verbose
    pushd work
    python3 -c 'import primitiv; dev = primitiv.devices.Naive()'
    popd
    pip3 uninstall -y primitiv

    pip3 install --user dist/primitiv-*.tar.gz --verbose
    pushd work
    python3 -c 'import primitiv; dev = primitiv.devices.Naive()'
    popd
    pip3 uninstall -y primitiv

    # test installing by "./setup.py install"
    ./setup.py install
    pushd work
    python3 -c 'import primitiv; dev = primitiv.devices.Naive()'
    popd
    pip3 uninstall -y primitiv
else
    pushd primitiv-core
    cmake .
    make
    make install
    popd
    ./setup.py build --no-build-core-library
    ./setup.py test --no-build-core-library

    # test installing by "./setup.py install"
    ./setup.py install --no-build-core-library
    pushd work
    python3 -c 'import primitiv; dev = primitiv.devices.Naive()'
    popd
    pip3 uninstall -y primitiv
fi

