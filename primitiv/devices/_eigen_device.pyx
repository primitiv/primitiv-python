from primitiv._device cimport Device


cdef class Eigen(Device):
    """Creates a Eigen object.

    """

    def __init__(self, rng_seed = None):
        """Creates a Eigen object.

        :param rng_seed: The seed value of internal random number generator.
        :type rng_seed: int or None

        """
        if self.wrapped is not NULL:
            raise TypeError("__init__() has already been called.")
        if rng_seed is None:
            self.wrapped = new CppEigen()
        else:
            self.wrapped = new CppEigen(<unsigned> rng_seed)

        Device.register_wrapper(self.wrapped, self)

    def __dealloc__(self):
        if self.wrapped is not NULL:
            del self.wrapped
            self.wrapped = NULL
