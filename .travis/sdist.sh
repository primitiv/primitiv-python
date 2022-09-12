#!/bin/bash
set -xe

if [ "${TRAVIS_BRANCH}" = "develop" ]; then
  PRIMITIV_PYTHON_BUILD_NUMBER="dev${TRAVIS_BUILD_NUMBER}"
else
  PRIMITIV_PYTHON_BUILD_NUMBER="${TRAVIS_BUILD_NUMBER}"
fi

export PRIMITIV_PYTHON_BUILD_NUMBER

pip install cython numpy scikit-build
wget -q "http://bitbucket.org/eigen/eigen/get/3.3.4.tar.bz2" -O eigen-downloaded.tar.gz
mkdir ./eigen-downloaded
tar xf ./eigen-downloaded.tar.gz --strip-components=1 -C ./eigen-downloaded
${TRAVIS_BUILD_DIR}/setup.py sdist --bundle-core-library --bundle-eigen-headers ./eigen-downloaded
pip install ${TRAVIS_BUILD_DIR}/dist/primitiv-*.tar.gz
mkdir ./work
pushd ./work
python -c "import primitiv; dev = primitiv.devices.Eigen()"
popd
