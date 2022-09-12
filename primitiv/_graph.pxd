from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from primitiv._device cimport CppDevice
from primitiv._shape cimport CppShape
from primitiv._tensor cimport CppTensor


cdef extern from "primitiv/core/graph.h" nogil:
    cdef cppclass CppNode "primitiv::Node":
        CppNode(CppNode &&src) except +
        CppNode() except +
        bool valid() except +
        CppGraph &graph() except +
        unsigned operator_id() except +
        unsigned value_id() except +
        CppShape shape() except +
        CppDevice &device() except +
        float to_float() except +
        vector[float] to_vector() except +
        vector[unsigned] argmax(unsigned dim) except +
        vector[unsigned] argmin(unsigned dim) except +
        void backward() except +


cdef extern from "primitiv/core/graph.h" nogil:
    cdef cppclass CppGraph "primitiv::Graph":
        CppGraph() except +
        void clear() except +
        const CppTensor &forward(const CppNode &node) except +
        void backward(const CppNode &node) except +
        CppShape get_shape(const CppNode &node) except +
        CppDevice &get_device(const CppNode &node) except +
        string dump(const string &format) except +
        unsigned num_operators() except +


cdef class Node:
    cdef CppNode wrapped


cdef class Graph:
    cdef CppGraph *wrapped
    cdef object __weakref__
    @staticmethod
    cdef void register_wrapper(CppGraph *ptr, Graph wrapper)
    @staticmethod
    cdef Graph get_wrapper(CppGraph *ptr)


cdef inline Node wrapNode(CppNode wrapped):
    cdef Node node = Node.__new__(Node)
    node.wrapped = wrapped
    return node
