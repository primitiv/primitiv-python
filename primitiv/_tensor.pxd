from libcpp.vector cimport vector
from libcpp cimport bool

from primitiv._device cimport CppDevice
from primitiv._shape cimport CppShape


cdef extern from "primitiv/core/tensor.h" nogil:
    cdef cppclass CppTensor "primitiv::Tensor":
        CppTensor(CppTensor &&src) except +
        CppTensor() except +
        bool valid() except +
        CppShape shape() except +
        CppDevice &device() except +
        const void *data() except +
        float to_float() except +
        vector[float] to_vector() except +
        vector[unsigned] argmax(unsigned dim) except +
        vector[unsigned] argmin(unsigned dim) except +
        void reset(float k) except +
        # void reset_by_array(const float *values) except +
        void reset_by_vector(const vector[float] &values) except +
        CppTensor reshape(const CppShape &new_shape) except +
        CppTensor flatten() except +
        CppTensor &inplace_multiply_const(float k) except +
        CppTensor &inplace_add(CppTensor &x) except +
        CppTensor &inplace_subtract(CppTensor &x) except +


cdef class Tensor:
    cdef CppTensor *wrapped
    cdef bool del_required
    cdef object __weakref__
    @staticmethod
    cdef void register_wrapper(CppTensor *ptr, Tensor wrapper)
    @staticmethod
    cdef Tensor get_wrapper(CppTensor *ptr)
    @staticmethod
    cdef Tensor get_wrapper_with_new(CppTensor *ptr)
