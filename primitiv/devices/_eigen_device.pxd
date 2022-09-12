from primitiv._device cimport CppDevice, Device


cdef extern from "primitiv/devices/eigen/device.h":
    cdef cppclass CppEigen "primitiv::devices::Eigen" (CppDevice):
        CppEigen() except +
        CppEigen(unsigned rng_seed) except +


cdef class Eigen(Device):
    pass
