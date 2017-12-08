from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map as cppmap
from libcpp cimport bool

from primitiv._device cimport CppDevice
from primitiv._parameter cimport CppParameter


cdef extern from "primitiv/model.h":
    cdef cppclass CppModel "primitiv::Model":
        CppModel() except +
        void load(string &path, bool with_stats, CppDevice *device) except +
        void save(string &path, bool with_stats) except +
        void add(string &name, CppParameter &param) except +
        void add(string &name, CppModel &model) except +
        CppParameter &get_parameter(string &name) except +
        CppParameter &get_parameter(vector[string] &names) except +
        CppModel &get_submodel(string &name) except +
        CppModel &get_submodel(vector[string] &names) except +
        cppmap[vector[string], CppParameter *] get_all_parameters() except +
        cppmap[vector[string], CppParameter *] get_trainable_parameters() except +


cdef class Model:
    cdef CppModel *wrapped
    cdef object __weakref__
    cdef object added
    @staticmethod
    cdef void register_wrapper(CppModel *ptr, Model wrapper)
    @staticmethod
    cdef Model get_wrapper(CppModel *ptr)

    # NOTE(vbkaisetsu)
    # Model is always created with `new`, so `del_required` is not used.
    # cdef bool del_required
