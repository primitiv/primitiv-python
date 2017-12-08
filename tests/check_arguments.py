from primitiv import Device
from primitiv import Graph
from primitiv import Parameter
from primitiv import Shape
from primitiv import initializers as I
from primitiv import functions as F
from primitiv.devices import Naive

import numpy as np
import unittest


class ArgumentTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.device = Naive()
        self.graph = Graph()
        Device.set_default(self.device)
        Graph.set_default(self.graph)
        self.ndarray_data = [
            np.array([
                [ 1, 2, 3],
                [ 4, 5, 6],
                [ 7, 8, 9],
                [10,11,12],
            ], np.float32),
            np.array([
                [13,14,15],
                [16,17,18],
                [19,20,21],
                [22,23,24],
            ], np.float32),
        ]
        self.list_data = [
             1.0,  4.0,  7.0, 10.0,  2.0,  5.0,  8.0, 11.0,  3.0,  6.0,  9.0, 12.0,
            13.0, 16.0, 19.0, 22.0, 14.0, 17.0, 20.0, 23.0, 15.0, 18.0, 21.0, 24.0,
        ]

    def tearDown(self):
        pass

    def test_functions_input_argument(self):
        # list[ndarray] w/o shape
        x = F.input(self.ndarray_data)
        self.assertEqual(x.to_list(), self.list_data)
        self.assertEqual(x.shape(), Shape([4, 3], 2))

        # ndarray w/o shape
        x = F.input(self.ndarray_data[0])
        self.assertEqual(x.to_list(), self.list_data[:12])
        self.assertEqual(x.shape(), Shape([4, 3], 1))

        # list[float] w/o shape
        self.assertRaises(TypeError, lambda: F.input(self.list_data))

        # list[float] w/ shape
        x = F.raw_input(Shape([4, 3], 2), self.list_data)
        self.assertEqual(x.to_list(), self.list_data)
        self.assertEqual(x.shape(), Shape([4, 3], 2))

    def test_Parameter_argument(self):
        # no argument
        p = Parameter()
        self.assertFalse(p.valid())

        # shape w/ Initializer
        p = Parameter(Shape([4, 3]), I.Constant(1))
        self.assertEqual(p.shape(), Shape([4, 3]))
        self.assertEqual(p.value.to_list(), [1] * 12)
