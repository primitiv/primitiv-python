#ifndef PYTHON_PRIMITIV_NODE_FUNC_H_
#define PYTHON_PRIMITIV_NODE_FUNC_H_

#include <primitiv/functions.h>
#include <primitiv/graph.h>

namespace python_primitiv_node {

using namespace primitiv;

inline Node func_node_pos(const Node &x) {
    return +x;
}

inline Node func_node_neg(const Node &x) {
    return -x;
}

inline Node func_node_add(const Node &x, float k) {
    return x + k;
}

inline Node func_node_add(float k, const Node &x) {
    return k + x;
}

inline Node func_node_add(const Node &a, const Node &b) {
    return a + b;
}

inline Node func_node_sub(const Node &x, float k) {
    return x - k;
}

inline Node func_node_sub(float k, const Node &x) {
    return k - x;
}

inline Node func_node_sub(const Node &a, const Node &b) {
    return a - b;
}

inline Node func_node_mul(const Node &x, float k) {
    return x * k;
}

inline Node func_node_mul(float k, const Node &x) {
    return k * x;
}

inline Node func_node_mul(const Node &a, const Node &b) {
    return a * b;
}

inline Node func_node_div(const Node &x, float k) {
    return x / k;
}

inline Node func_node_div(float k, const Node &x) {
    return k / x;
}

inline Node func_node_div(const Node &a, const Node &b) {
    return a / b;
}

}  // namespace python_primitiv_node

#endif  // PYTHON_PRIMITIV_NODE_FUNC_H_
