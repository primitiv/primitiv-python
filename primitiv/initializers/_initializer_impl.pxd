from primitiv._initializer cimport CppInitializer, Initializer


cdef extern from "primitiv/core/initializer_impl.h":
    cdef cppclass CppConstant "primitiv::initializers::Constant" (CppInitializer):
        CppConstant(float k)

    cdef cppclass CppUniform "primitiv::initializers::Uniform" (CppInitializer):
        CppUniform(float lower, float upper)

    cdef cppclass CppNormal "primitiv::initializers::Normal" (CppInitializer):
        CppNormal(float mean, float sd)

    cdef cppclass CppIdentity "primitiv::initializers::Identity" (CppInitializer):
        CppIdentity()

    cdef cppclass CppXavierUniform "primitiv::initializers::XavierUniform" (CppInitializer):
        CppXavierUniform(float scale)

    cdef cppclass CppXavierNormal "primitiv::initializers::XavierNormal" (CppInitializer):
        CppXavierNormal(float scale)

    cdef cppclass CppXavierUniformConv2D "primitiv::initializers::XavierUniformConv2D" (CppInitializer):
        CppXavierUniformConv2D(float scale)

    cdef cppclass CppXavierNormalConv2D "primitiv::initializers::XavierNormalConv2D" (CppInitializer):
        CppXavierNormalConv2D(float scale)


cdef class Constant(Initializer):
    pass

cdef class Uniform(Initializer):
    pass

cdef class Normal(Initializer):
    pass

cdef class Identity(Initializer):
    pass

cdef class XavierUniform(Initializer):
    pass

cdef class XavierNormal(Initializer):
    pass

cdef class XavierUniformConv2D(Initializer):
    pass

cdef class XavierNormalConv2D(Initializer):
    pass
