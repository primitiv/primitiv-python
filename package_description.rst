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

Build and install primitiv without CUDA and OpenCL::

    pip3 install primitiv

Build and install primitiv with CUDA and/or OpenCL support::

    # Enable only CUDA
    pip3 install primitiv --global-option --enable-cuda

    # Enable both CUDA and OpenCL
    pip3 install primitiv --global-option --enable-cuda --global-option --enable-opencl

Notes
-----

According to the ``manylinux1`` policy described in
`PEP 513 <https://www.python.org/dev/peps/pep-0513/>`_, binary packages
should depend only on an extremely limited set of external shared libraries.
Most users may install primitiv with CUDA or OpenCL that is not supported in
``manylinux1`` policy, so we provide only a source pacakge for now.

Resources
---------

* `Homepage <https://github.com/primitiv/primitiv-python>`_
* `Examples <https://github.com/primitiv/primitiv-python/tree/develop/examples>`_
