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
- CMake (3.1.0 or later)
- scikit-build (0.6.1 or later)
- (optional) CUDA (7.5 or later)
- (optional) OpenCL (1.2 or later) and OpenCL C++ binding v2

Install required packages::

    pip3 install numpy cython cmake scikit-build

Build and install primitiv without CUDA and OpenCL::

    pip3 install primitiv

Build and install primitiv with CUDA and/or OpenCL support::

    # Enable only CUDA
    pip3 install primitiv --global-option --enable-cuda

    # Enable both CUDA and OpenCL
    pip3 install primitiv --global-option --enable-cuda --global-option --enable-opencl

``--enable-eigen`` flag that enables ``Eigen`` backend is added by default in
the package contained in PyPI. To disable the Eigen backend, use
``--disable-eigen`` flag. Note that Eigen is bundled with the package contained
in PyPI.


Notes
-----

We are providing only a source pacakge for now, and ``pip`` command
downloads the source package and builds it before installing.
This is mainly because of keeping compatibility with the ``manylinux1`` standard
described in `PEP 513 <https://www.python.org/dev/peps/pep-0513/>`_
while maintaining supports of non-standard backends such as CUDA/OpenCL.


Resources
---------

* `Official repository <https://github.com/primitiv/primitiv-python>`_
* `Examples <https://github.com/primitiv/primitiv-python/tree/develop/examples>`_
