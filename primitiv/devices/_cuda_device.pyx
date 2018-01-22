from primitiv._device cimport Device


cdef class CUDA(Device):

    def __init__(self, unsigned device_id, rng_seed = None):
        """Creates a new CUDA device.

        :param device_id: ID of the physical GPU.
        :type device_id: int
        :param rng_seed: The seed value of the random number generator (default: None).
        :type rng_seed: int or None
        :raises RuntimeError: if this class does not support the specified device.

        """
        if self.wrapped is not NULL:
            raise TypeError("__init__() has already been called.")
        if rng_seed is None:
            self.wrapped = new CppCUDA(device_id)
        else:
            self.wrapped = new CppCUDA(device_id, <unsigned> rng_seed)
        Device.register_wrapper(self.wrapped, self)

    def __dealloc__(self):
        if self.wrapped is not NULL:
            del self.wrapped
            self.wrapped = NULL

    @staticmethod
    def num_devices():
        """Retrieves the number of active hardwares.

        :return: Number of active hardwares.
        :rtype: int

        """
        return CppCUDA.num_devices()

    @staticmethod
    def check_support(unsigned device_id):
        """Checks whether the device corresponding to the specified ID is
        supported.

        :param device_id: Device ID to check.
        :type: int
        :return: True if this class supports the specified device,
                 False otherwise.
        :rtype: bool

        """
        return CppCUDA.check_support(device_id)
