cdef extern from "primitiv/core/device.h":
    cdef cppclass CppDevice "primitiv::Device":
        void dump_description() except +


cdef class Device:
    cdef CppDevice *wrapped
    cdef object __weakref__
    @staticmethod
    cdef void register_wrapper(CppDevice *ptr, Device wrapper)
    @staticmethod
    cdef Device get_wrapper(CppDevice *ptr)
