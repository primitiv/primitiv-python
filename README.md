[![python](https://img.shields.io/badge/python-3.5-blue.svg)](https://www.python.org/)
[![backend](https://img.shields.io/badge/backend-CPU%2c%20CUDA%2c%20OpenCL-blue.svg)](README.md)
[![os](https://img.shields.io/badge/os-Ubuntu%2c%20Debian%2c%20Fedora%2c%20OSX-blue.svg)](https://travis-ci.org/odashi/primitiv)
[![license](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Build Status (master)](https://img.shields.io/travis/primitiv/primitiv-python/master.svg?label=build+%28master%29)](https://travis-ci.org/primitiv/primitiv-python)
[![Build Status (develop)](https://img.shields.io/travis/primitiv/primitiv-python/develop.svg?label=build+%28develop%29)](https://travis-ci.org/primitiv/primitiv-python)
[![PyPI version](https://badge.fury.io/py/primitiv.svg)](https://pypi.python.org/pypi/primitiv)


Python Frontend of primitiv
===========================

This frontend is compatible with **primitiv core library 0.4.x**.


Dependencies
------------

* Python3 (3.5 or later)
* NumPy (1.11.0 or later)
* Cython (0.27 or later)
* CMake (3.1.0 or later)
* scikit-build (0.6.1 or later)
* (optional) CUDA (7.5 or later)
* (optional) Eigen (Eigen 3.3.0  or later)
* (optional) OpenCL (1.2 or later) and OpenCL C++ binding v2


Getting Started
---------------

### Using `pip`

```
$ pip3 install numpy cython cmake scikit-build [--user]
$ pip3 install primitiv [--user]
                        [--global-option --enable-cuda]
                        [--global-option --enable-opencl]
```

To enable CUDA and/or OpenCL support, specify `--enable-cuda` or
`--enable-opencl` with `--global-option` flag.

`--enable-eigen` flag that enables `Eigen` backend is added by default in the
package contained in PyPI. To disable the Eigen backend, use `--disable-eigen`
flag. Note that Eigen is bundled with the package contained in PyPI.

We are providing only a source pacakge for now, and `pip` command
downloads the source package and builds it before installing.
This is mainly because of keeping compatibility with the `manylinux1` standard
described in [PEP 513](https://www.python.org/dev/peps/pep-0513/)
while maintaining supports of non-standard backends such as CUDA/OpenCL.

### Installing from source

```
$ git clone https://github.com/primitiv/primitiv-python
$ cd primitiv-python
$ git submodule update --init
$ pip3 install -r requirements.txt
$ python3 ./setup.py build [--enable-cuda] [--enable-eigen] [--enable-opencl]
$ python3 ./setup.py test [--enable-cuda] [--enable-eigen] [--enable-opencl] # (optional)
$ python3 ./setup.py install [--user] [--enable-cuda] [--enable-eigen] [--enable-opencl]
```

*primitiv-python* repository contains the core library as a git submodule.
Note that you have to update the working tree of the core library manually by
`git submodule update` after you run `git pull` or `git checkout` commands.


Resources
---------

* [C++ core library of primitiv](https://github.com/primitiv/primitiv)
* [Examples/tutorials](https://github.com/primitiv/primitiv-python/tree/develop/examples)
