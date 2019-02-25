#ifndef PRIMITIV_PYTHON_PY_OPTIMIZER_H_
#define PRIMITIV_PYTHON_PY_OPTIMIZER_H_

#include <primitiv/core/optimizer.h>
#include <iostream>

__PYX_EXTERN_C int primitiv_python_optimizer_get_configs(
                        PyObject *obj,
                        std::unordered_map<std::string, unsigned> &uint_configs,
                        std::unordered_map<std::string, float> &float_configs);
__PYX_EXTERN_C int primitiv_python_optimizer_set_configs(PyObject *obj,
                        const std::unordered_map<std::string, unsigned> &uint_configs,
                        const std::unordered_map<std::string, float> &float_configs);
__PYX_EXTERN_C int primitiv_python_optimizer_configure_parameter(
                        PyObject *obj,
                        primitiv::Parameter &param);
__PYX_EXTERN_C int primitiv_python_optimizer_update_parameter(
                        PyObject *obj,
                        float scale,
                        primitiv::Parameter &param);

namespace primitiv {

namespace python {

class PyOptimizer : public primitiv::Optimizer {

public:
  PyOptimizer(PyObject *obj) : obj_(obj) {}

  void get_configs(std::unordered_map<std::string, unsigned> &uint_configs,
                   std::unordered_map<std::string, float> &float_configs) const override {
    int ret = primitiv_python_optimizer_get_configs(obj_, uint_configs, float_configs);
    if (ret == -1) {
      // NOTE(vbkaisetsu): This is just a trigger of throwing an error.
      // This message is not passed to Python.
      PRIMITIV_THROW_ERROR("error: get_configs");
    }
    Optimizer::get_configs(uint_configs, float_configs);
  }

  void set_configs(const std::unordered_map<std::string, unsigned> &uint_configs,
                   const std::unordered_map<std::string, float> &float_configs) override {
    Optimizer::set_configs(uint_configs, float_configs);
    int ret = primitiv_python_optimizer_set_configs(obj_, uint_configs, float_configs);
    if (ret == -1) {
      // NOTE(vbkaisetsu): This is just a trigger of throwing an error.
      // This message is not passed to Python.
      PRIMITIV_THROW_ERROR("error: set_configs");
    }
  }

  void configure_parameter(primitiv::Parameter &param) override {
    int ret = primitiv_python_optimizer_configure_parameter(obj_, param);
    if (ret == -1) {
      // NOTE(vbkaisetsu): This is just a trigger of throwing an error.
      // This message is not passed to Python.
      PRIMITIV_THROW_ERROR("error: configure_parameter");
    }
  }

  void update_parameter(float scale, primitiv::Parameter &param) override {
    int ret = primitiv_python_optimizer_update_parameter(obj_, scale, param);
    if (ret == -1) {
      // NOTE(vbkaisetsu): This is just a trigger of throwing an error.
      // This message is not passed to Python.
      PRIMITIV_THROW_ERROR("error: update_parameter");
    }
  }

private:
  PyObject *obj_;
};

}  // namespace python

}  // namespace primitiv

#endif  // PRIMITIV_PYTHON_PY_OPTIMIZER_H_
