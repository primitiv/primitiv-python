from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

from primitiv._device cimport CppDevice
from primitiv._graph cimport CppGraph, CppNode
from primitiv._tensor cimport CppTensor
from primitiv._shape cimport CppShape
from primitiv._parameter cimport CppParameter

cdef extern from "function_template_wrapper.h" namespace "python_primitiv":

    CppNode Node_input_vector(const CppShape &shape, const vector[float] &data, CppDevice &dev, CppGraph &g) except +
    CppNode Node_input_vector(const CppShape &shape, const vector[float] &data, CppDevice &dev) except +
    CppNode Node_parameter(CppParameter &param, CppGraph &g) except +
    CppNode Node_parameter(CppParameter &param) except +
    CppNode Node_sum(const CppNode &x, unsigned dim) except +
    CppNode Node_sum_container(const vector[CppNode] &xs) except +
    CppNode Node_mean(const CppNode &x, unsigned dim) except +
    CppNode Node_mean_container(const vector[CppNode] &xs) except +

    CppTensor Tensor_input_vector(const CppShape &shape, const vector[float] &data, CppDevice &dev) except +
    CppTensor Tensor_parameter(CppParameter &param) except +
    CppTensor Tensor_sum(const CppTensor &x, unsigned dim) except +
    CppTensor Tensor_sum_container(const vector[CppTensor] &xs) except +
    CppTensor Tensor_mean(const CppTensor &x, unsigned dim) except +
    CppTensor Tensor_mean_container(const vector[CppTensor] &xs) except +


cdef extern from "primitiv/functions.h":
    Var func_copy "primitiv::functions::copy" [Var](const Var &x, CppDevice &dev) except +
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
    Var func_ipow "primitiv::functions::ipow" [Var](const Var &x, int k) except +
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
    Var func_broadcast "primitiv::functions::broadcast" [Var](const Var &x, unsigned dim, unsigned size) except +
    Var func_logsumexp "primitiv::functions::logsumexp" [Var](const Var &x, unsigned dim) except +
    Var func_log_softmax "primitiv::functions::log_softmax" [Var](const Var &x, unsigned dim) except +
    Var func_softmax "primitiv::functions::softmax" [Var](const Var &x, unsigned dim) except +
    Var func_softmax_cross_entropy "primitiv::functions::softmax_cross_entropy" [Var](const Var &x, const Var &t, unsigned dim) except +
    Var func_softmax_cross_entropy "primitiv::functions::softmax_cross_entropy" [Var](const Var &x, const vector[unsigned] &ids, unsigned dim) except +
    Var func_stop_gradient "primitiv::functions::stop_gradient" [Var](const Var &x) except +
    CppNode func_constant "primitiv::functions::constant" (const CppShape &shape, float k, CppDevice &dev, CppGraph &g) except +
    CppNode func_zeros "primitiv::functions::zeros" (const CppShape &shape, CppDevice &dev, CppGraph &g) except +
    CppNode func_ones "primitiv::functions::ones" (const CppShape &shape, CppDevice &dev, CppGraph &g) except +
    CppNode func_identity "primitiv::functions::identity" (unsigned size, CppDevice &dev, CppGraph &g) except +
    Var func_constant "primitiv::functions::constant" [Var](const CppShape &shape, float k, CppDevice &dev) except +
    Var func_zeros "primitiv::functions::zeros" [Var](const CppShape &shape, CppDevice &dev) except +
    Var func_ones "primitiv::functions::ones" [Var](const CppShape &shape, CppDevice &dev) except +
    Var func_identity "primitiv::functions::identity" [Var](unsigned size, CppDevice &dev) except +
    Var func_dropout "primitiv::functions::dropout" [Var](const Var &x, float rate, bool enabled) except +


cdef extern from "primitiv/functions.h":
    Var func_batch_sum "primitiv::functions::batch::sum" [Var](const Var &x) except +
    Var func_batch_mean "primitiv::functions::batch::mean" [Var](const Var &x) except +
    Var func_batch_normalize "primitiv::functions::batch::normalize" [Var](const Var &x) except +


cdef extern from "primitiv/functions.h":

    CppNode func_random_bernoulli "primitiv::functions::random::bernoulli" (const CppShape &shape, float p, CppDevice &dev, CppGraph &g) except +
    Var func_random_bernoulli "primitiv::functions::random::bernoulli" [Var](const CppShape &shape, float p, CppDevice &dev) except +
    CppNode func_random_uniform "primitiv::functions::random::uniform" (const CppShape &shape, float lower, float upper, CppDevice &dev, CppGraph &g) except +
    Var func_random_uniform "primitiv::functions::random::uniform" [Var](const CppShape &shape, float lower, float upper, CppDevice &dev) except +
    CppNode func_random_normal "primitiv::functions::random::normal" (const CppShape &shape, float mean, float sd, CppDevice &dev, CppGraph &g) except +
    Var func_random_normal "primitiv::functions::random::normal" [Var](const CppShape &shape, float mean, float sd, CppDevice &dev) except +
    CppNode func_random_log_normal "primitiv::functions::random::log_normal" (const CppShape &shape, float mean, float sd, CppDevice &dev, CppGraph &g) except +
    Var func_random_log_normal "primitiv::functions::random::log_normal" [Var](const CppShape &shape, float mean, float sd, CppDevice &dev) except +
    CppNode func_random_gumbel "primitiv::functions::random::gumbel" (const CppShape &shape, float mu, float beta, CppDevice &dev, CppGraph &g) except +
    Var func_random_gumbel "primitiv::functions::random::gumbel" [Var](const CppShape &shape, float mu, float beta, CppDevice &dev) except +
