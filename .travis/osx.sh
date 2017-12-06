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
    ./setup.py build && ./setup.py test

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
else
    git clone https://github.com/primitiv/primitiv.git libprimitiv
    pushd libprimitiv
    cmake .
    make && make install
    popd
    ./setup.py build && ./setup.py test
fi

# test installing by "./setup.py install"
./setup.py install
pushd work
python3 -c 'import primitiv; dev = primitiv.devices.Naive()'
popd
pip3 uninstall -y primitiv
