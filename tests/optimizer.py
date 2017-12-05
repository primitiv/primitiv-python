from primitiv import Device
from primitiv import devices as D
from primitiv import initializers as I
from primitiv import Model
from primitiv import optimizers as O
from primitiv import Parameter

import unittest


class TestModel(Model):
    pass


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
        p = Parameter([5], I.Constant(0))
        optimizer = O.Adam()
        optimizer.set_weight_decay(1e-6)
        optimizer.set_gradient_clipping(5)
        optimizer.add(model)
        optimizer.add(p)
