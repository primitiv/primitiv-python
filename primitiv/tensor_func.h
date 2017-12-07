#ifndef PYTHON_PRIMITIV_TENSOR_FUNC_H_
#define PYTHON_PRIMITIV_TENSOR_FUNC_H_

#include <primitiv/functions.h>
#include <primitiv/tensor.h>

namespace python_primitiv_tensor {

inline primitiv::Tensor func_tensor_pos(const primitiv::Tensor &x) {
    return +x;
}

inline primitiv::Tensor func_tensor_neg(const primitiv::Tensor &x) {
    return -x;
}

inline primitiv::Tensor func_tensor_add(const primitiv::Tensor &x, float k) {
    return x + k;
}

inline primitiv::Tensor func_tensor_add(float k, const primitiv::Tensor &x) {
    return k + x;
}

inline primitiv::Tensor func_tensor_add(const primitiv::Tensor &a, const primitiv::Tensor &b) {
    return a + b;
}

inline primitiv::Tensor func_tensor_sub(const primitiv::Tensor &x, float k) {
    return x - k;
}

inline primitiv::Tensor func_tensor_sub(float k, const primitiv::Tensor &x) {
    return k - x;
}

inline primitiv::Tensor func_tensor_sub(const primitiv::Tensor &a, const primitiv::Tensor &b) {
    return a - b;
}

inline primitiv::Tensor func_tensor_mul(const primitiv::Tensor &x, float k) {
    return x * k;
}

inline primitiv::Tensor func_tensor_mul(float k, const primitiv::Tensor &x) {
    return k * x;
}

inline primitiv::Tensor func_tensor_mul(const primitiv::Tensor &a, const primitiv::Tensor &b) {
    return a * b;
}

inline primitiv::Tensor func_tensor_div(const primitiv::Tensor &x, float k) {
    return x / k;
}

inline primitiv::Tensor func_tensor_div(float k, const primitiv::Tensor &x) {
    return k / x;
}

inline primitiv::Tensor func_tensor_div(const primitiv::Tensor &a, const primitiv::Tensor &b) {
    return a / b;
}

inline void func_tensor_imul(primitiv::Tensor &tensor, float k) {
    tensor *= k;
}

inline void func_tensor_iadd(primitiv::Tensor &tensor, const primitiv::Tensor &x) {
    tensor += x;
}

inline void func_tensor_isub(primitiv::Tensor &tensor, const primitiv::Tensor &x) {
    tensor -= x;
}

}  // namespace python_primitiv_tensor

#endif  // PYTHON_PRIMITIV_TENSOR_FUNC_H_
