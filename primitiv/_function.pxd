from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

from primitiv._device cimport CppDevice
from primitiv._graph cimport CppGraph, CppNode
from primitiv._tensor cimport CppTensor
from primitiv._shape cimport CppShape
from primitiv._parameter cimport CppParameter


cdef extern from "primitiv/core/functions.h":
    CppTensor func_input_tensor "primitiv::functions::input_tensor" (const CppShape &shape, const vector[float] &data, CppDevice *dev) except +
    CppNode func_input_node "primitiv::functions::input_node" (const CppShape &shape, const vector[float] &data, CppDevice *dev, CppGraph *g) except +
    CppTensor func_parameter_tensor "primitiv::functions::parameter_tensor" (CppParameter &param) except +
    CppNode func_parameter_node "primitiv::functions::parameter_node" (CppParameter &param, CppGraph *g) except +
    Var func_copy "primitiv::functions::copy" [Var](const Var &x, CppDevice *dev) except +
    Var func_pick "primitiv::functions::pick" [Var](const Var &x, const vector[unsigned] &ids, unsigned dim) except +
    Var func_slice "primitiv::functions::slice" [Var](const Var &x, unsigned dim, unsigned lower, unsigned upper) except +
    Var func_concat "primitiv::functions::concat" [Var](const vector[Var] &xs, unsigned dim) except +
    Var func_reshape "primitiv::functions::reshape" [Var](const Var &x, const CppShape &new_shape) except +
    Var func_flatten "primitiv::functions::flatten" [Var](const Var &x) except +
    Var func_transpose "primitiv::functions::transpose" [Var](const Var &x) except +
    Var func_matmul "primitiv::functions::matmul" [Var](const Var &a, const Var &b) except +
    Var func_sqrt "primitiv::functions::sqrt" [Var](const Var &x) except +
    Var func_exp "primitiv::functions::exp" [Var](const Var &x) except +
    Var func_log "primitiv::functions::log" [Var](const Var &x) except +
    Var func_pown "primitiv::functions::pown" [Var](const Var &x, int k) except +
    Var func_pow "primitiv::functions::pow" [Var](const Var &x, float k) except +
    Var func_pow "primitiv::functions::pow" [Var](float x, const Var &k) except +
    Var func_pow "primitiv::functions::pow" [Var](const Var &x, const Var &k) except +
    Var func_tanh "primitiv::functions::tanh" [Var](const Var &x) except +
    Var func_sigmoid "primitiv::functions::sigmoid" [Var](const Var &x) except +
    Var func_softplus "primitiv::functions::softplus" [Var](const Var &x) except +
    Var func_sin "primitiv::functions::sin" [Var](const Var &x) except +
    Var func_cos "primitiv::functions::cos" [Var](const Var &x) except +
    Var func_tan "primitiv::functions::tan" [Var](const Var &x) except +
    Var func_relu "primitiv::functions::relu" [Var](const Var &x) except +
    Var func_lrelu "primitiv::functions::lrelu" [Var](const Var &x) except +
    Var func_prelu "primitiv::functions::prelu" [Var](const Var &x, float a) except +
    Var func_elu "primitiv::functions::elu" [Var](const Var &x, float a) except +
    Var func_selu "primitiv::functions::selu" [Var](const Var &x, float a, float s) except +
    CppNode func_sum "primitiv::functions::sum" (const vector[CppNode] &xs) except +
    CppTensor func_sum "primitiv::functions::sum" (const vector[CppTensor] &xs) except +
    Var func_sum "primitiv::functions::sum" [Var](const Var &x, unsigned dim) except +
    CppNode func_mean "primitiv::functions::mean" (const vector[CppNode] &xs) except +
    CppTensor func_mean "primitiv::functions::mean" (const vector[CppTensor] &xs) except +
    Var func_mean "primitiv::functions::mean" [Var](const Var &x, unsigned dim) except +
    Var func_broadcast "primitiv::functions::broadcast" [Var](const Var &x, unsigned dim, unsigned size) except +
    Var func_logsumexp "primitiv::functions::logsumexp" [Var](const Var &x, unsigned dim) except +
    Var func_log_softmax "primitiv::functions::log_softmax" [Var](const Var &x, unsigned dim) except +
    Var func_softmax "primitiv::functions::softmax" [Var](const Var &x, unsigned dim) except +
    Var func_softmax_cross_entropy "primitiv::functions::softmax_cross_entropy" [Var](const Var &x, const Var &t, unsigned dim) except +
    Var func_softmax_cross_entropy "primitiv::functions::softmax_cross_entropy" [Var](const Var &x, const vector[unsigned] &ids, unsigned dim) except +
    Var func_stop_gradient "primitiv::functions::stop_gradient" [Var](const Var &x) except +
    Var func_conv2d "primitiv::functions::conv2d" [Var](const Var &x, const Var &w, unsigned padding0, unsigned padding1, unsigned stride0, unsigned stride1, unsigned dilation0, unsigned dilation1) except +
    Var func_max_pool2d "primitiv::functions::max_pool2d" [Var](const Var &x, unsigned window0, unsigned window1, unsigned padding0, unsigned padding1, unsigned stride0, unsigned stride1) except +

    CppTensor func_constant_tensor "primitiv::functions::constant_tensor" (const CppShape &shape, float k, CppDevice *dev) except +
    CppNode func_constant_node "primitiv::functions::constant_node" (const CppShape &shape, float k, CppDevice *dev, CppGraph *g) except +
    CppTensor func_zeros_tensor "primitiv::functions::zeros_tensor" (const CppShape &shape, CppDevice *dev) except +
    CppNode func_zeros_node "primitiv::functions::zeros_node" (const CppShape &shape, CppDevice *dev, CppGraph *g) except +
    CppTensor func_ones_tensor "primitiv::functions::ones_tensor" (const CppShape &shape, CppDevice *dev) except +
    CppNode func_ones_node "primitiv::functions::ones_node" (const CppShape &shape, CppDevice *dev, CppGraph *g) except +
    CppTensor func_identity_tensor "primitiv::functions::identity_tensor" (unsigned size, CppDevice *dev) except +
    CppNode func_identity_node "primitiv::functions::identity_node" (unsigned size, CppDevice *dev, CppGraph *g) except +
    Var func_dropout "primitiv::functions::dropout" [Var](const Var &x, float rate, bool enabled) except +

    Var func_positive "primitiv::functions::positive" [Var](const Var &x) except +
    Var func_negative "primitiv::functions::negative" [Var](const Var &x) except +
    Var func_add "primitiv::functions::add" [Var](const Var &x, float k) except +
    Var func_add "primitiv::functions::add" [Var](float k, const Var &x) except +
    Var func_add "primitiv::functions::add" [Var](const Var &a, const Var &b) except +
    Var func_subtract "primitiv::functions::subtract" [Var](const Var &x, float k) except +
    Var func_subtract "primitiv::functions::subtract" [Var](float k, const Var &x) except +
    Var func_subtract "primitiv::functions::subtract" [Var](const Var &a, const Var &b) except +
    Var func_multiply "primitiv::functions::multiply" [Var](const Var &x, float k) except +
    Var func_multiply "primitiv::functions::multiply" [Var](float k, const Var &x) except +
    Var func_multiply "primitiv::functions::multiply" [Var](const Var &a, const Var &b) except +
    Var func_divide "primitiv::functions::divide" [Var](const Var &x, float k) except +
    Var func_divide "primitiv::functions::divide" [Var](float k, const Var &x) except +
    Var func_divide "primitiv::functions::divide" [Var](const Var &a, const Var &b) except +


cdef extern from "primitiv/core/functions.h":
    Var func_batch_sum "primitiv::functions::batch::sum" [Var](const Var &x) except +
    Var func_batch_mean "primitiv::functions::batch::mean" [Var](const Var &x) except +
    Var func_batch_normalize "primitiv::functions::batch::normalize" [Var](const Var &x) except +


cdef extern from "primitiv/core/functions.h":

    CppNode func_random_bernoulli_node "primitiv::functions::random::bernoulli_node" (const CppShape &shape, float p, CppDevice *dev, CppGraph *g) except +
    CppTensor func_random_bernoulli_tensor "primitiv::functions::random::bernoulli_tensor" (const CppShape &shape, float p, CppDevice *dev) except +
    CppNode func_random_uniform_node "primitiv::functions::random::uniform_node" (const CppShape &shape, float lower, float upper, CppDevice *dev, CppGraph *g) except +
    CppTensor func_random_uniform_tensor "primitiv::functions::random::uniform_tensor" (const CppShape &shape, float lower, float upper, CppDevice *dev) except +
    CppNode func_random_normal_node "primitiv::functions::random::normal_node" (const CppShape &shape, float mean, float sd, CppDevice *dev, CppGraph *g) except +
    CppTensor func_random_normal_tensor "primitiv::functions::random::normal_tensor" (const CppShape &shape, float mean, float sd, CppDevice *dev) except +
    CppNode func_random_log_normal_node "primitiv::functions::random::log_normal_node" (const CppShape &shape, float mean, float sd, CppDevice *dev, CppGraph *g) except +
    CppTensor func_random_log_normal_tensor "primitiv::functions::random::log_normal_tensor" (const CppShape &shape, float mean, float sd, CppDevice *dev) except +
    CppNode func_random_gumbel_node "primitiv::functions::random::gumbel_node" (const CppShape &shape, float mu, float beta, CppDevice *dev, CppGraph *g) except +
    CppTensor func_random_gumbel_tensor "primitiv::functions::random::gumbel_tensor" (const CppShape &shape, float mu, float beta, CppDevice *dev) except +
