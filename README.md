[![python](https://img.shields.io/badge/python-3.5-blue.svg)](https://www.python.org/)
[![backend](https://img.shields.io/badge/backend-CPU%2c%20CUDA%2c%20OpenCL-blue.svg)](README.md)
[![os](https://img.shields.io/badge/os-Ubuntu%2c%20Debian%2c%20Fedora%2c%20OSX-blue.svg)](https://travis-ci.org/odashi/primitiv)
[![license](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Build Status (master)](https://img.shields.io/travis/primitiv/primitiv-python/master.svg?label=build+%28master%29)](https://travis-ci.org/primitiv/primitiv-python)
[![Build Status (develop)](https://img.shields.io/travis/primitiv/primitiv-python/develop.svg?label=build+%28develop%29)](https://travis-ci.org/primitiv/primitiv-python)
[![PyPI version](https://badge.fury.io/py/primitiv.svg)](https://pypi.python.org/pypi/primitiv)

Python frontend of primitiv
=================================

Dependency
---------------------------------

* C++ primitiv
* Python 3 (3.5 or later)
* NumPy (1.11.0 or later)
* Cython (0.27 or later)

How to install?
---------------------------------

1. Install [primitiv core library](http://github.com/primitiv/primitiv):

```
$ cmake .. [options]
$ make
$ sudo make install
```

2. Install NumPy and Cython with Python 3

```
$ sudo apt install python3-numpy
$ sudo pip3 install cython
```

Currently, Cython 0.27 is not contained in Debian and Ubuntu repositories.

3. Run the following commands in python-primitiv directory:

```
$ python3 ./setup.py build [--enable-cuda] [--enable-opencl]
$ python3 ./setup.py test  # (optional)
$ sudo python3 ./setup.py install [--enable-cuda] [--enable-opencl]
```

You also can use `LIBRARY_PATH` and `CPLUS_INCLUDE_PATH` depending on your environment.

To install CUDA and/or OpenCL support, run setup script with `--enable-DEVICE` option.
