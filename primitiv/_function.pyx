from libcpp.vector cimport vector
from libcpp cimport bool

from primitiv._device cimport Device
from primitiv._shape cimport Shape, normShape
from primitiv._tensor cimport Tensor, CppTensor
from primitiv._graph cimport Graph, wrapNode, CppNode, Node
from primitiv._parameter cimport Parameter

from utils cimport ndarrays_to_vector, get_cpp_device, get_cpp_graph

cimport numpy as np
import numpy as np


class functions:

    @staticmethod
    def raw_input(shape, vector[float] data, Device device = None, Graph graph = None):
        if device is None:
            device = Device.get_default()
        if graph is None:
            graph = Graph.get_default()
        return wrapNode(func_input_node(normShape(shape).wrapped, data,
                                        get_cpp_device(device), get_cpp_graph(graph)))

    # NOTE(vbkaisetsu)
    # This function takes an np.ndarray or a list of np.ndarray
    # instead of a vector.
    @staticmethod
    def input(data, Device device = None, Graph graph = None):
        # NOTE(vbkaisetsu, odashi):
        # In this function, we don't check whether each ndarray is empty
        # (i.e., it doesn't have any elements) or not.
        # When the ndarray contains no element, its shape becomes (0,),
        # and primitiv.Shape will reject the shape and raises an exception.
        # In addition, we also don't check whether each ndarray has the same shape or not.
        # This condition will be checked in ndarrays_to_vector().
        if isinstance(data, np.ndarray):
            data = [data]
        if isinstance(data, list):
            if len(data) == 0:
                raise TypeError("`data` contains no item.")
            if not isinstance(data[0], np.ndarray):
                raise TypeError("`data` contains other objects than numpy.ndarray.")
            shape = Shape(data[0].shape, len(data))
        else:
            raise TypeError("`data` has incorrect type.")
        return functions.raw_input(shape, ndarrays_to_vector(data), device, graph)


    @staticmethod
    def parameter(Parameter param, Graph graph = None):
        if graph is None:
            graph = Graph.get_default()
        return wrapNode(func_parameter_node(param.wrapped[0], get_cpp_graph(graph)))

    @staticmethod
    def copy(Node x, Device device = None):
        if device is None:
            device = Device.get_default()
        return wrapNode(func_copy(x.wrapped, get_cpp_device(device)))

    @staticmethod
    def pick(Node x, vector[unsigned] ids, unsigned dim):
        return wrapNode(func_pick(x.wrapped, ids, dim))

    @staticmethod
    def slice(Node x, unsigned dim, unsigned lower, unsigned upper):
        return wrapNode(func_slice(x.wrapped, dim, lower, upper))

    @staticmethod
    def concat(xs, unsigned dim):
        cdef vector[CppNode] vec
        cdef Node x
        for x in xs:
            vec.push_back(x.wrapped)
        return wrapNode(func_concat(vec, dim))

    @staticmethod
    def reshape(Node x, new_shape):
        return wrapNode(func_reshape(x.wrapped, normShape(new_shape).wrapped))

    @staticmethod
    def flatten(Node x):
        return wrapNode(func_flatten(x.wrapped))

    @staticmethod
    def transpose(Node x):
        return wrapNode(func_transpose(x.wrapped))

    @staticmethod
    def matmul(Node a, Node b):
        return wrapNode(func_matmul(a.wrapped, b.wrapped))

    @staticmethod
    def sqrt(Node x):
        return wrapNode(func_sqrt(x.wrapped))

    @staticmethod
    def exp(Node x):
        return wrapNode(func_exp(x.wrapped))

    @staticmethod
    def log(Node x):
        return wrapNode(func_log(x.wrapped))

    @staticmethod
    def pow(x, k):
        if isinstance(x, Node) and isinstance(k, int) and -0x80000000 <= k <= 0x7fffffff:
            return wrapNode(func_pown((<Node> x).wrapped, <int> k))
        elif isinstance(x, Node) and isinstance(k, (int, float)):
            return wrapNode(func_pow((<Node> x).wrapped, <float> k))
        elif isinstance(x, (int, float)) and isinstance(k, Node):
            return wrapNode(func_pow(<float> x, (<Node> k).wrapped))
        elif isinstance(x, Node) and isinstance(k, Node):
            return wrapNode(func_pow((<Node> x).wrapped, (<Node> k).wrapped))
        else:
            raise TypeError("`x` or `k` has incorrect type.")

    @staticmethod
    def tanh(Node x):
        return wrapNode(func_tanh(x.wrapped))

    @staticmethod
    def sigmoid(Node x):
        return wrapNode(func_sigmoid(x.wrapped))

    @staticmethod
    def softplus(Node x):
        return wrapNode(func_softplus(x.wrapped))

    @staticmethod
    def sin(Node x):
        return wrapNode(func_sin(x.wrapped))

    @staticmethod
    def cos(Node x):
        return wrapNode(func_cos(x.wrapped))

    @staticmethod
    def tan(Node x):
        return wrapNode(func_tan(x.wrapped))

    @staticmethod
    def relu(Node x):
        return wrapNode(func_relu(x.wrapped))

    @staticmethod
    def lrelu(Node x):
        return wrapNode(func_lrelu(x.wrapped))

    @staticmethod
    def prelu(Node x, float a):
        return wrapNode(func_prelu(x.wrapped, a))

    @staticmethod
    def elu(Node x, float a):
        return wrapNode(func_elu(x.wrapped, a))

    @staticmethod
    def selu(Node x, float a, float s):
        return wrapNode(func_selu(x.wrapped, a, s))

    @staticmethod
    def sum(x, dim = None):
        cdef vector[CppNode] xs
        cdef Node node
        if isinstance(x, list):
            for node in x:
                xs.push_back(node.wrapped)
            return wrapNode(func_sum(xs))
        else:
            return wrapNode(func_sum((<Node> x).wrapped, <unsigned> dim))

    @staticmethod
    def mean(x, dim = None):
        cdef vector[CppNode] xs
        cdef Node node
        if isinstance(x, list):
            for node in x:
                xs.push_back(node.wrapped)
            return wrapNode(func_mean(xs))
        else:
            return wrapNode(func_mean((<Node> x).wrapped, <unsigned> dim))

    @staticmethod
    def broadcast(Node x, unsigned dim, unsigned size):
        return wrapNode(func_broadcast(x.wrapped, dim, size))

    @staticmethod
    def logsumexp(Node x, unsigned dim):
        return wrapNode(func_logsumexp(x.wrapped, dim))

    @staticmethod
    def log_softmax(Node x, unsigned dim):
        return wrapNode(func_log_softmax(x.wrapped, dim))

    @staticmethod
    def softmax(Node x, unsigned dim):
        return wrapNode(func_softmax(x.wrapped, dim))

    @staticmethod
    def softmax_cross_entropy(Node x, t, unsigned dim):
        if isinstance(t, Node):
            return wrapNode(func_softmax_cross_entropy(x.wrapped, (<Node> t).wrapped, dim))
        elif isinstance(t, list):
            return wrapNode(func_softmax_cross_entropy(x.wrapped, <vector[unsigned]> t, dim))
        else:
            raise TypeError("`t` has incorrect type.")

    @staticmethod
    def stop_gradient(Node x):
        return wrapNode(func_stop_gradient(x.wrapped))

    @staticmethod
    def conv2d(Node x, Node w,
               unsigned padding0, unsigned padding1,
               unsigned stride0, unsigned stride1,
               unsigned dilation0, unsigned dilation1):
        return wrapNode(func_conv2d(x.wrapped, w.wrapped,
                                    padding0, padding1,
                                    stride0, stride1,
                                    dilation0, dilation1))

    @staticmethod
    def max_pool2d(Node x,
                   unsigned window0, unsigned window1,
                   unsigned padding0, unsigned padding1,
                   unsigned stride0, unsigned stride1):
        return wrapNode(func_max_pool2d(x.wrapped,
                                        window0, window1,
                                        padding0, padding1,
                                        stride0, stride1))

    @staticmethod
    def constant(shape, float k, Device device = None, Graph graph = None):
        if device is None:
            device = Device.get_default()
        if graph is None:
            graph = Graph.get_default()
        return wrapNode(func_constant_node(normShape(shape).wrapped, k,
                                           get_cpp_device(device), get_cpp_graph(graph)))

    @staticmethod
    def zeros(shape, Device device = None, Graph graph = None):
        if device is None:
            device = Device.get_default()
        if graph is None:
            graph = Graph.get_default()
        return wrapNode(func_zeros_node(normShape(shape).wrapped,
                                        get_cpp_device(device), get_cpp_graph(graph)))

    @staticmethod
    def ones(shape, Device device = None, Graph graph = None):
        if device is None:
            device = Device.get_default()
        if graph is None:
            graph = Graph.get_default()
        return wrapNode(func_ones_node(normShape(shape).wrapped,
                                       get_cpp_device(device), get_cpp_graph(graph)))

    @staticmethod
    def identity(unsigned size, Device device = None, Graph graph = None):
        if device is None:
            device = Device.get_default()
        if graph is None:
            graph = Graph.get_default()
        return wrapNode(func_identity_node(size, get_cpp_device(device), get_cpp_graph(graph)))

    class batch:
        @staticmethod
        def sum(Node x):
            return wrapNode(func_batch_sum[CppNode](x.wrapped))

        @staticmethod
        def mean(Node x):
            return wrapNode(func_batch_mean[CppNode](x.wrapped))

        @staticmethod
        def normalize(Node x):
            return wrapNode(func_batch_normalize[CppNode](x.wrapped))

    class random:
        @staticmethod
        def bernoulli(shape, float p, Device device = None, Graph graph = None):
            if device is None:
                device = Device.get_default()
            if graph is None:
                graph = Graph.get_default()
            return wrapNode(func_random_bernoulli_node(normShape(shape).wrapped, p,
                                                       get_cpp_device(device), get_cpp_graph(graph)))

        @staticmethod
        def uniform(shape, float lower, float upper, Device device = None, Graph graph = None):
            if device is None:
                device = Device.get_default()
            if graph is None:
                graph = Graph.get_default()
            return wrapNode(func_random_uniform_node(normShape(shape).wrapped, lower, upper,
                                                     get_cpp_device(device), get_cpp_graph(graph)))

        @staticmethod
        def normal(shape, float mean, float sd, Device device = None, Graph graph = None):
            if device is None:
                device = Device.get_default()
            if graph is None:
                graph = Graph.get_default()
            return wrapNode(func_random_normal_node(normShape(shape).wrapped, mean, sd,
                                                    get_cpp_device(device), get_cpp_graph(graph)))

        @staticmethod
        def log_normal(shape, float mean, float sd, Device device = None, Graph graph = None):
            if device is None:
                device = Device.get_default()
            if graph is None:
                graph = Graph.get_default()
            return wrapNode(func_random_log_normal_node(normShape(shape).wrapped, mean, sd,
                                                        get_cpp_device(device), get_cpp_graph(graph)))

        @staticmethod
        def gumbel(shape, float mu, float beta, Device device = None, Graph graph = None):
            if device is None:
                device = Device.get_default()
            if graph is None:
                graph = Graph.get_default()
            return wrapNode(func_random_gumbel_node(normShape(shape).wrapped, mu, beta,
                                                    get_cpp_device(device), get_cpp_graph(graph)))

    @staticmethod
    def dropout(Node x, float rate, bool enabled):
        return wrapNode(func_dropout(x.wrapped, rate, enabled))


class tensor_functions:

    @staticmethod
    def raw_input(shape, vector[float] data, Device device = None):
        if device is None:
            device = Device.get_default()
        return Tensor.get_wrapper_with_new(new CppTensor(func_input_tensor(normShape(shape).wrapped, data,
                                                                           get_cpp_device(device))))

    # NOTE(vbkaisetsu)
    # This function takes an np.ndarray or a list of np.ndarray
    # instead of a vector.
    @staticmethod
    def input(data, Device device = None):
        # NOTE(vbkaisetsu, odashi):
        # In this function, we don't check whether each ndarray is empty
        # (i.e., it doesn't have any elements) or not.
        # When the ndarray contains no element, its shape becomes (0,),
        # and primitiv.Shape will reject the shape and raises an exception.
        # In addition, we also don't check whether each ndarray has the same shape or not.
        # This condition will be checked in ndarrays_to_vector().
        if isinstance(data, np.ndarray):
            data = [data]
        if isinstance(data, list):
            if len(data) == 0:
                raise TypeError("`data` contains no item.")
            if not isinstance(data[0], np.ndarray):
                raise TypeError("`data` contains other objects than numpy.ndarray.")
            shape = Shape(data[0].shape, len(data))
        else:
            raise TypeError("`data` has incorrect type.")
        return tensor_functions.raw_input(shape, ndarrays_to_vector(data), device)

    @staticmethod
    def parameter(Parameter param):
        return Tensor.get_wrapper_with_new(new CppTensor(func_parameter_tensor(param.wrapped[0])))

    @staticmethod
    def copy(Tensor x, Device device = None):
        if device is None:
            device = Device.get_default()
        return Tensor.get_wrapper_with_new(new CppTensor(func_copy(x.wrapped[0], get_cpp_device(device))))

    @staticmethod
    def pick(Tensor x, vector[unsigned] ids, unsigned dim):
        return Tensor.get_wrapper_with_new(new CppTensor(func_pick(x.wrapped[0], ids, dim)))

    @staticmethod
    def slice(Tensor x, unsigned dim, unsigned lower, unsigned upper):
        return Tensor.get_wrapper_with_new(new CppTensor(func_slice(x.wrapped[0], dim, lower, upper)))

    @staticmethod
    def concat(xs, unsigned dim):
        cdef vector[CppTensor] vec
        cdef Tensor x
        for x in xs:
            vec.push_back(x.wrapped[0])
        return Tensor.get_wrapper_with_new(new CppTensor(func_concat(vec, dim)))

    @staticmethod
    def reshape(Tensor x, new_shape):
        return Tensor.get_wrapper_with_new(new CppTensor(func_reshape(x.wrapped[0], normShape(new_shape).wrapped)))

    @staticmethod
    def flatten(Tensor x):
        return Tensor.get_wrapper_with_new(new CppTensor(func_flatten(x.wrapped[0])))

    @staticmethod
    def transpose(Tensor x):
        return Tensor.get_wrapper_with_new(new CppTensor(func_transpose(x.wrapped[0])))

    @staticmethod
    def matmul(Tensor a, Tensor b):
        return Tensor.get_wrapper_with_new(new CppTensor(func_matmul(a.wrapped[0], b.wrapped[0])))

    @staticmethod
    def sqrt(Tensor x):
        return Tensor.get_wrapper_with_new(new CppTensor(func_sqrt(x.wrapped[0])))

    @staticmethod
    def exp(Tensor x):
        return Tensor.get_wrapper_with_new(new CppTensor(func_exp(x.wrapped[0])))

    @staticmethod
    def log(Tensor x):
        return Tensor.get_wrapper_with_new(new CppTensor(func_log(x.wrapped[0])))

    @staticmethod
    def pow(x, k):
        if isinstance(x, Tensor) and isinstance(k, int) and -0x80000000 <= k <= 0x7fffffff:
            return Tensor.get_wrapper_with_new(new CppTensor(func_pown((<Tensor> x).wrapped[0], <int> k)))
        elif isinstance(x, Tensor) and isinstance(k, (int, float)):
            return Tensor.get_wrapper_with_new(new CppTensor(func_pow((<Tensor> x).wrapped[0], <float> k)))
        elif isinstance(x, (int, float)) and isinstance(k, Tensor):
            return Tensor.get_wrapper_with_new(new CppTensor(func_pow(<float> x, (<Tensor> k).wrapped[0])))
        elif isinstance(x, Tensor) and isinstance(k, Tensor):
            return Tensor.get_wrapper_with_new(new CppTensor(func_pow((<Tensor> x).wrapped[0], (<Tensor> k).wrapped[0])))
        else:
            raise TypeError("`x` or `k` has incorrect type.")

    @staticmethod
    def tanh(Tensor x):
        return Tensor.get_wrapper_with_new(new CppTensor(func_tanh(x.wrapped[0])))

    @staticmethod
    def sigmoid(Tensor x):
        return Tensor.get_wrapper_with_new(new CppTensor(func_sigmoid(x.wrapped[0])))

    @staticmethod
    def softplus(Tensor x):
        return Tensor.get_wrapper_with_new(new CppTensor(func_softplus(x.wrapped[0])))

    @staticmethod
    def sin(Tensor x):
        return Tensor.get_wrapper_with_new(new CppTensor(func_sin(x.wrapped[0])))

    @staticmethod
    def cos(Tensor x):
        return Tensor.get_wrapper_with_new(new CppTensor(func_cos(x.wrapped[0])))

    @staticmethod
    def tan(Tensor x):
        return Tensor.get_wrapper_with_new(new CppTensor(func_tan(x.wrapped[0])))

    @staticmethod
    def relu(Tensor x):
        return Tensor.get_wrapper_with_new(new CppTensor(func_relu(x.wrapped[0])))

    @staticmethod
    def lrelu(Tensor x):
        return Tensor.get_wrapper_with_new(new CppTensor(func_lrelu(x.wrapped[0])))

    @staticmethod
    def prelu(Tensor x, float a):
        return Tensor.get_wrapper_with_new(new CppTensor(func_prelu(x.wrapped[0], a)))

    @staticmethod
    def elu(Tensor x, float a):
        return Tensor.get_wrapper_with_new(new CppTensor(func_elu(x.wrapped[0], a)))

    @staticmethod
    def selu(Tensor x, float a, float s):
        return Tensor.get_wrapper_with_new(new CppTensor(func_selu(x.wrapped[0], a, s)))

    @staticmethod
    def sum(x, dim = None):
        cdef vector[CppTensor] xs
        cdef Tensor t
        if isinstance(x, list):
            for t in x:
                xs.push_back(t.wrapped[0])
            return Tensor.get_wrapper_with_new(new CppTensor(func_sum(xs)))
        else:
            return Tensor.get_wrapper_with_new(new CppTensor(func_sum((<Tensor> x).wrapped[0], <unsigned> dim)))

    @staticmethod
    def mean(x, dim = None):
        cdef vector[CppTensor] xs
        cdef Tensor t
        if isinstance(x, list):
            for t in x:
                xs.push_back(t.wrapped[0])
            return Tensor.get_wrapper_with_new(new CppTensor(func_mean(xs)))
        else:
            return Tensor.get_wrapper_with_new(new CppTensor(func_mean((<Tensor> x).wrapped[0], <unsigned> dim)))

    @staticmethod
    def broadcast(Tensor x, unsigned dim, unsigned size):
        return Tensor.get_wrapper_with_new(new CppTensor(func_broadcast(x.wrapped[0], dim, size)))

    @staticmethod
    def logsumexp(Tensor x, unsigned dim):
        return Tensor.get_wrapper_with_new(new CppTensor(func_logsumexp(x.wrapped[0], dim)))

    @staticmethod
    def log_softmax(Tensor x, unsigned dim):
        return Tensor.get_wrapper_with_new(new CppTensor(func_log_softmax(x.wrapped[0], dim)))

    @staticmethod
    def softmax(Tensor x, unsigned dim):
        return Tensor.get_wrapper_with_new(new CppTensor(func_softmax(x.wrapped[0], dim)))

    @staticmethod
    def softmax_cross_entropy(Tensor x, t, unsigned dim):
        if isinstance(t, Tensor):
            return Tensor.get_wrapper_with_new(new CppTensor(func_softmax_cross_entropy(x.wrapped[0], (<Tensor> t).wrapped[0], dim)))
        elif isinstance(t, list):
            return Tensor.get_wrapper_with_new(new CppTensor(func_softmax_cross_entropy(x.wrapped[0], <vector[unsigned]> t, dim)))
        else:
            raise TypeError("`t` has incorrect type.")

    @staticmethod
    def stop_gradient(Tensor x):
        return Tensor.get_wrapper_with_new(new CppTensor(func_stop_gradient(x.wrapped[0])))

    @staticmethod
    def conv2d(Tensor x, Tensor w,
               unsigned padding0, unsigned padding1,
               unsigned stride0, unsigned stride1,
               unsigned dilation0, unsigned dilation1):
        return Tensor.get_wrapper_with_new(new CppTensor(func_conv2d(x.wrapped[0], w.wrapped[0],
                                                                     padding0, padding1,
                                                                     stride0, stride1,
                                                                     dilation0, dilation1)))

    @staticmethod
    def max_pool2d(Tensor x,
                   unsigned window0, unsigned window1,
                   unsigned padding0, unsigned padding1,
                   unsigned stride0, unsigned stride1):
        return Tensor.get_wrapper_with_new(new CppTensor(func_max_pool2d(x.wrapped[0],
                                                                         window0, window1,
                                                                         padding0, padding1,
                                                                         stride0, stride1)))

    @staticmethod
    def constant(shape, float k, Device device = None):
        if device is None:
            device = Device.get_default()
        return Tensor.get_wrapper_with_new(new CppTensor(func_constant_tensor(normShape(shape).wrapped, k,
                                                                              get_cpp_device(device))))

    @staticmethod
    def zeros(shape, Device device = None):
        if device is None:
            device = Device.get_default()
        return Tensor.get_wrapper_with_new(new CppTensor(func_zeros_tensor(normShape(shape).wrapped,
                                                                           get_cpp_device(device))))

    @staticmethod
    def ones(shape, Device device = None):
        if device is None:
            device = Device.get_default()
        return Tensor.get_wrapper_with_new(new CppTensor(func_ones_tensor(normShape(shape).wrapped,
                                                                          get_cpp_device(device))))

    @staticmethod
    def identity(unsigned size, Device device = None):
        if device is None:
            device = Device.get_default()
        return Tensor.get_wrapper_with_new(new CppTensor(func_identity_tensor(size, get_cpp_device(device))))

    class batch:
        @staticmethod
        def sum(Tensor x):
            return Tensor.get_wrapper_with_new(new CppTensor(func_batch_sum[CppTensor](x.wrapped[0])))

        @staticmethod
        def mean(Tensor x):
            return Tensor.get_wrapper_with_new(new CppTensor(func_batch_mean[CppTensor](x.wrapped[0])))

        @staticmethod
        def normalize(Tensor x):
            return Tensor.get_wrapper_with_new(new CppTensor(func_batch_normalize[CppTensor](x.wrapped[0])))

    class random:
        @staticmethod
        def bernoulli(shape, float p, Device device = None):
            if device is None:
                device = Device.get_default()
            return Tensor.get_wrapper_with_new(new CppTensor(func_random_bernoulli_tensor(normShape(shape).wrapped, p,
                                                                                          get_cpp_device(device))))

        @staticmethod
        def uniform(shape, float lower, float upper, Device device = None):
            if device is None:
                device = Device.get_default()
            return Tensor.get_wrapper_with_new(new CppTensor(func_random_uniform_tensor(normShape(shape).wrapped, lower, upper,
                                                                                        get_cpp_device(device))))

        @staticmethod
        def normal(shape, float mean, float sd, Device device = None):
            if device is None:
                device = Device.get_default()
            return Tensor.get_wrapper_with_new(new CppTensor(func_random_normal_tensor(normShape(shape).wrapped, mean, sd,
                                                                                       get_cpp_device(device))))

        @staticmethod
        def log_normal(shape, float mean, float sd, Device device = None):
            if device is None:
                device = Device.get_default()
            return Tensor.get_wrapper_with_new(new CppTensor(func_random_log_normal_tensor(normShape(shape).wrapped, mean, sd,
                                                                                           get_cpp_device(device))))

        @staticmethod
        def gumbel(shape, float mu, float beta, Device device = None):
            if device is None:
                device = Device.get_default()
            return Tensor.get_wrapper_with_new(new CppTensor(func_random_gumbel_tensor(normShape(shape).wrapped, mu, beta,
                                                                                       get_cpp_device(device))))

    @staticmethod
    def dropout(Tensor x, float rate, bool enabled):
        return Tensor.get_wrapper_with_new(new CppTensor(func_dropout(x.wrapped[0], rate, enabled)))
