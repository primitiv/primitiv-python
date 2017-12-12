[![python](https://img.shields.io/badge/python-3.5-blue.svg)](https://www.python.org/)
[![backend](https://img.shields.io/badge/backend-CPU%2c%20CUDA%2c%20OpenCL-blue.svg)](README.md)
[![os](https://img.shields.io/badge/os-Ubuntu%2c%20Debian%2c%20Fedora%2c%20OSX-blue.svg)](https://travis-ci.org/odashi/primitiv)
[![license](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Build Status (master)](https://img.shields.io/travis/primitiv/primitiv-python/master.svg?label=build+%28master%29)](https://travis-ci.org/primitiv/primitiv-python)
[![Build Status (develop)](https://img.shields.io/travis/primitiv/primitiv-python/develop.svg?label=build+%28develop%29)](https://travis-ci.org/primitiv/primitiv-python)
[![PyPI version](https://badge.fury.io/py/primitiv.svg)](https://pypi.python.org/pypi/primitiv)

Python Frontend of primitiv
===========================

Dependencies
------------

* Python 3 (3.5 or later)
* NumPy (1.11.0 or later)
* Cython (0.27 or later)
* scikit-build (0.6.1 or later, only for building)
* (optional) CUDA (7.5 or later)
* (optional) OpenCL (1.2 or later) and OpenCL C++ binding v2

Getting Started
---------------

### Automatic Install using `pip`

To install primitiv without CUDA and OpenCL, run the following commands:

```
$ pip3 install numpy cython scikit-build [--user]
$ pip3 install primitiv [--user]
```

To enable CUDA and/or OpenCL support, specify `--enable-cuda` or
`--enable-opencl` with `--global-option` flag of `pip` like the following
example:

```
$ pip3 install primitiv --global-option --enable-cuda \
                        --global-option --enable-opncl
```

### Compiling Step by Step

1. Install NumPy, Cython and scikit-build with Python 3

```
$ sudo pip3 install numpy cython scikit-build
```

2. Run the following commands in `primitiv-python` directory:

```
$ git submodule update --init
$ python3 ./setup.py build [--enable-cuda] [--enable-opencl]
$ python3 ./setup.py test [--enable-cuda] [--enable-opencl] # (optional)
$ python3 ./setup.py install [--user] [--enable-cuda] [--enable-opencl]
```

To enable CUDA and/or OpenCL support, run setup script with `--enable-DEVICE` option.

primitiv-python repository contains the core library as a git submodule.
Note that you have to update the working tree of the core library manually by
`git submodule update` after you run `git pull` or `git checkout` commands.

Resources
---------

* [C++ core library of primitiv](https://github.com/primitiv/primitiv)
* [Examples](https://github.com/primitiv/primitiv-python/tree/develop/examples)
