from libcpp.pair cimport pair
from libc.stdint cimport uintptr_t

from primitiv._device cimport Device
from primitiv._parameter cimport Parameter
from primitiv.config cimport pystr_to_cppstr, cppstr_to_pystr
from utils cimport get_cpp_device

from weakref import WeakValueDictionary
import weakref

# NOTE(vbkaisetsu):
# This is used for holding python instances related to C++.
# Without this variable, python instances are always created when C++ class
# instances are returned from functions.
# It means that users can not compare instances by using "is" operator.
cdef object py_primitiv_model_weak_dict = WeakValueDictionary()


cdef class Model:

    def __cinit__(self):
        self.wrapped = new CppModel()
        Model.register_wrapper(self.wrapped, self)
        self.added = []

    def __dealloc__(self):
        if self.wrapped is not NULL:
            del self.wrapped
            self.wrapped = NULL

    def load(self, str path, bool with_stats = True, Device device = None):
        """Loads all parameters from a file.

        :param path: Path of the file.
        :type path: str
        :param with_stats: Whether or not to load all additional statistics (default: ``True``).
        :type with_stats: bool
        :param device: Device object to manage parameters (default: ``None``).
        :type device: bool or None

        """
        if device is None:
            device = Device.get_default()
        self.wrapped.load(pystr_to_cppstr(path), with_stats,
                          get_cpp_device(device))

    def save(self, str path, bool with_stats = True):
        """Saves all parameters to a file.

        :param path: Path of the file.
        :type path: str
        :param with_stats: Whether or not to save all additional statistics (default: ``True``).
        :type with_stats: bool

        """
        self.wrapped.save(pystr_to_cppstr(path), with_stats)

    def add(self, str name, arg):
        """Registers a new parameter or a new submodel.

        :param name: Name of the submodel.
        :type name: str
        :param arg: Parameter or submodel to register.
        :type arg: primitiv.Parameter or primitiv.Model

        ``name`` should not be overlapped with all registered parameters and
        submodels.

        This function does not modify attribute information of this object.
        To set ``arg`` as an attribule, use ``__setattr__`` instead.

        """
        if isinstance(arg, Parameter):
            self.wrapped.add(pystr_to_cppstr(name), (<Parameter> arg).wrapped[0])
        elif isinstance(arg, Model):
            self.wrapped.add(pystr_to_cppstr(name), (<Model> arg).wrapped[0])
        else:
            raise TypeError("Argument 'arg' has incorrect type (Parameter or Model)")
        self.added.append(arg)

    def __setattr__(self, key, value):
        """Set attribute

        If Parameter or Model is set, add(key, value) is additionally
        called to register a new parameter. Otherwise, a value is
        normally set to this model.

        """
        if isinstance(value, Parameter) and value not in self.added:
            self.add(key, value)
        if isinstance(value, Model) and value not in self.added:
            self.add(key, value)
        self.__dict__[key] = value

    def __delattr__(self, key):
        # NOTE(vbkaisetsu): __delattr__ is not called when the parent object is deleted.
        item = self.__dict__[key]
        if isinstance(item, Parameter) or isinstance(item, Model):
            raise TypeError("Parameter and Model are not deletable.")
        del self.__dict__[key]

    def __getitem__(self, key):
        """Retrieves a parameter or a model in this model.

        :param key: Name hierarchy of the submodel or parameter.
        :type key: str or tuple[str]

        The following example retrieves ``sub3`` in ``sub2`` in ``sub1`` model
        in the model instance ``m``:

            >>> m = MyModel()
            >>> :::
            >>> m["sub1", "sub2", "sub3"]

        """
        cdef vector[string] names
        if isinstance(key, str):
            names.push_back(pystr_to_cppstr(key))
        elif isinstance(key, tuple):
            for name in key:
                names.push_back(pystr_to_cppstr(name))
        else:
            raise TypeError("Argument 'key' has incorrect type (str or tuple)")
        try:
            return Parameter.get_wrapper(&self.wrapped.get_parameter(names))
        except:
            try:
                return Model.get_wrapper(&self.wrapped.get_submodel(names))
            except:
                # NOTE(vbkaisetsu): DO NOT throw an exception here, because
                # error massages generated at above lines should not be shown.
                pass
        raise TypeError("'name' is not a name of neither parameter nor submodel")

    def get_all_parameters(self):
        """Retrieves all parameters in the model.

        :return: Dictionary of all parameters.
        :rtype: dict[str, primitiv.Parameter]

        """
        cdef pair[vector[string], CppParameter*] p
        result = {}
        for p in self.wrapped.get_all_parameters():
            result[tuple(cppstr_to_pystr(s) for s in p.first)] = Parameter.get_wrapper(p.second)
        return result

    def get_trainable_parameters(self):
        """Retrieves trainable parameters in the model.

        :return: Dictionary of trainable parameters.
        :rtype: dict[str, primitiv.Parameter]

        """
        cdef pair[vector[string], CppParameter*] p
        result = {}
        for p in self.wrapped.get_trainable_parameters():
            result[tuple(cppstr_to_pystr(s) for s in p.first)] = Parameter.get_wrapper(p.second)
        return result

    @staticmethod
    cdef void register_wrapper(CppModel *ptr, Model wrapper):
        if <uintptr_t> ptr in py_primitiv_model_weak_dict:
            raise ValueError("Attempted to register the same C++ object twice.")
        py_primitiv_model_weak_dict[<uintptr_t> ptr] = wrapper

    @staticmethod
    cdef Model get_wrapper(CppModel *ptr):
        return py_primitiv_model_weak_dict[<uintptr_t> ptr]
