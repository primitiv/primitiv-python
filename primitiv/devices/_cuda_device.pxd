from primitiv._device cimport CppDevice, Device


cdef extern from "primitiv/devices/cuda/device.h":
    cdef cppclass CppCUDA "primitiv::devices::CUDA" (CppDevice):
        CppCUDA(unsigned device_id) except +
        CppCUDA(unsigned device_id, unsigned rng_seed) except +
        @staticmethod
        unsigned num_devices() except +
        @staticmethod
        unsigned check_support(unsigned device_id) except +


cdef class CUDA(Device):
    pass
