primitiv: A Neural Network Toolkit. (Python frontend)
=====================================================

Features
--------

- Dynamic and incremental graph construction
- On-demand memory allocation
- Automatic minibatch broadcasting
- Mostly device-independent
- Simple usage

Install
-------

Prerequisites:

- Python 3 (3.5 or later)
- NumPy (1.11.0 or later)
- Cython (0.27 or later)
- scikit-build (0.6.1 or later, only for building)
- (optional) CUDA (7.5 or later)
- (optional) OpenCL (1.2 or later) and OpenCL C++ binding v2

Install dependencies::

    pip3 install numpy cython scikit-build

Install primitiv without CUDA and OpenCL::

    pip3 install primitiv

Install primitiv with CUDA and/or OpenCL support::

    # Enable only CUDA
    pip3 install primitiv --global-option --enable-cuda

    # Enable both CUDA and OpenCL
    pip3 install primitiv --global-option --enable-cuda --global-option --enable-opencl

Resources
---------

* `Homepage <https://github.com/primitiv/primitiv-python>`_
* `Examples <https://github.com/primitiv/primitiv-python/tree/develop/examples>`_
