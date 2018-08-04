from primitiv._device cimport CppDevice, Device


cdef extern from "primitiv/devices/naive/device.h":
    cdef cppclass CppNaive "primitiv::devices::Naive" (CppDevice):
        CppNaive() except +
        CppNaive(unsigned rng_seed) except +


cdef class Naive(Device):
    pass
