from primitiv import Device
from primitiv import devices as D
from primitiv import initializers as I
from primitiv import Model
from primitiv import optimizers as O
from primitiv import Parameter
from primitiv import tensor_operators as tF

import unittest


class TestModel(Model):
    def __init__(self):
        self.param = Parameter([1], I.Constant(0))
        self.param.gradient = tF.raw_input([1], [1])
        self.scan_attributes()


class Optimizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.device = D.Naive()
        Device.set_default(self.device)

    def tearDown(self):
        pass

    def test_optimizer_add(self):
        model = TestModel()
        p = Parameter([1], I.Constant(0))
        p.gradient = tF.raw_input([1], [1])
        optimizer = O.Adam()
        optimizer.set_weight_decay(1e-6)
        optimizer.set_gradient_clipping(5)
        optimizer.add(model)
        optimizer.add(p)
        self.assertEqual(p.gradient.to_list()[0], 1)
        self.assertEqual(model.param.gradient.to_list()[0], 1)
        optimizer.reset_gradients()
        self.assertEqual(p.gradient.to_list()[0], 0)
        self.assertEqual(model.param.gradient.to_list()[0], 0)
