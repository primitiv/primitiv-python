#!/bin/bash
set -xe

pip install twine
if [ "${BINARY_PACKAGE}" = "yes" ]; then
  twine upload -u "${PYPI_USERNAME}" -p "${PYPI_PASSWORD}" $TRAVIS_BUILD_DIR/wheelhouse/primitiv-*.whl;
else
  twine upload -u "${PYPI_USERNAME}" -p "${PYPI_PASSWORD}" $TRAVIS_BUILD_DIR/dist/primitiv-*.tar.gz;
fi
