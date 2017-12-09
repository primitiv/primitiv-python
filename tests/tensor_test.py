import random
import unittest

from primitiv import Shape
from primitiv import Tensor
from primitiv import tensor_functions as tF

import numpy as np
from . import test_utils


class TensorTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = test_utils.available_devices()

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_TensorTest_CheckInvalid(self):
        x = Tensor()
        self.assertFalse(x.valid())
        with self.assertRaises(RuntimeError):
            x.shape()
        with self.assertRaises(RuntimeError):
            x.device()
        with self.assertRaises(RuntimeError):
            x.to_float()
        with self.assertRaises(RuntimeError):
            x.to_list()
        with self.assertRaises(RuntimeError):
            x.to_ndarrays()

    def test_TensorTest_CheckNewScalarWithData(self):
        for dev in TensorTest.devices:
            x = tF.raw_input([], [1], dev)
            x_ndarray = [
                np.array([1]),
            ]
            self.assertTrue(x.valid())
            self.assertIs(dev, x.device())
            self.assertEqual(Shape(), x.shape())
            self.assertEqual([1], x.to_list())
            self.assertEqual(1.0, x.to_float())
            self.assertEqual(x_ndarray, x.to_ndarrays())

    def test_TensorTest_CheckNewMatrixWithData(self):
        for dev in TensorTest.devices:
            data = [1, 2, 3, 4, 5, 6]
            data_ndarray = [
                np.array([[1, 3, 5], [2, 4, 6]]),
            ]
            x = tF.raw_input([2, 3], data, dev)
            self.assertTrue(x.valid())
            self.assertIs(dev, x.device())
            self.assertEqual(Shape([2, 3]), x.shape())
            self.assertEqual(data, x.to_list())
            with self.assertRaises(RuntimeError):
                x.to_float()
            self.assertTrue(np.array_equal(data_ndarray, x.to_ndarrays()))

    def test_TensorTest_CheckNewMatrixMinibatchWithData(self):
        for dev in TensorTest.devices:
            data = [
                3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8,
                9, 7, 9, 3, 2, 3, 8, 4, 6, 2, 6, 4,
            ]
            data_ndarray = [
                np.array([[3, 4, 5], [1, 1, 9]]),
                np.array([[2, 5, 5], [6, 3, 8]]),
                np.array([[9, 9, 2], [7, 3, 3]]),
                np.array([[8, 6, 6], [4, 2, 4]]),
            ]
            x = tF.raw_input(Shape([2, 3], 4), data, dev)
            self.assertTrue(x.valid())
            self.assertIs(dev, x.device())
            self.assertEqual(Shape([2, 3], 4), x.shape())
            self.assertEqual(data, x.to_list())
            with self.assertRaises(RuntimeError):
                x.to_float()
            self.assertTrue(np.array_equal(data_ndarray, x.to_ndarrays()))

    def test_TensorTest_CheckCopyValidToNew(self):
        for dev in TensorTest.devices:
            print(dev)
            tmp = tF.raw_input(Shape([2], 3), [1, 2, 3, 4, 5, 6], dev)
            x = Tensor(tmp)
            self.assertTrue(x.valid())
            self.assertTrue(tmp.valid())
            self.assertEqual(Shape([2], 3), x.shape())
            self.assertEqual(Shape([2], 3), tmp.shape())
            self.assertEqual([1, 2, 3, 4, 5, 6], x.to_list())
            self.assertEqual([1, 2, 3, 4, 5, 6], tmp.to_list())

    def test_TensorTest_CheckCopyInvalidToNew(self):
        for dev in TensorTest.devices:
            tmp = Tensor()
            x = Tensor(tmp)
            self.assertFalse(x.valid())
            self.assertFalse(tmp.valid())

    def test_TesnorTest_CheckResetValuesByConstant(self):
        for dev in TensorTest.devices:
            x = tF.raw_input(Shape([2, 2], 2), [42] * 8, dev)
            self.assertEqual([42] * 8, x.to_list())

            x = tF.raw_input(Shape([2, 2], 2), [0] * 8, dev)
            x.reset(42)
            self.assertEqual([42] * 8, x.to_list())

            x = tF.raw_input(Shape([2, 2], 2), [123] * 8, dev)
            copied = Tensor(x)

            x.reset(42)
            self.assertEqual([42] * 8, x.to_list())
            self.assertEqual([123] * 8, copied.to_list())

    def test_TensorTest_CheckResetValuesByVector(self):
        for dev in TensorTest.devices:
            data = [1, 2, 3, 4, 5, 6, 7, 8]
            x = tF.raw_input(Shape([2, 2], 2), data, dev)
            self.assertEqual(data, x.to_list())

            data = [1, 2, 3, 4, 5, 6, 7, 8]
            x = tF.raw_input(Shape([2, 2], 2), [0] * 8, dev)
            x.reset_by_vector(data)
            self.assertEqual(data, x.to_list())

            data = [1, 2, 3, 4, 5, 6, 7, 8]
            x = tF.raw_input(Shape([2, 2], 2), [123] * 8, dev)
            copied = Tensor(x)

            x.reset_by_vector(data)
            self.assertEqual(data, x.to_list())
            self.assertEqual([123] * 8, copied.to_list())

    def test_TensorTest_InplaceMultiplyConst(self):
        for dev in TensorTest.devices:
            x_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            y_data = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
            x = tF.raw_input(Shape([2, 2], 3), x_data, dev)
            x *= 2
            self.assertEqual(y_data, x.to_list())
        for dev in TensorTest.devices:
            x_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            y_data = [.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
            x = tF.raw_input(Shape([2, 2], 3), x_data, dev)
            x *= .5
            self.assertEqual(y_data, x.to_list())

    def test_TensorTest_CheckInplaceAddNN(self):
        for dev in TensorTest.devices:
            a_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            b_data = [0, -1, -2, -3, -3, -4, -5, -6, -6, -7, -8, -9]
            y_data = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
            a = tF.raw_input(Shape([2, 2], 3), a_data, dev)
            b = tF.raw_input(Shape([2, 2], 3), b_data, dev)
            a += b
            self.assertEqual(y_data, a.to_list())

    def test_TensorTest_CheckInplaceAdd1N(self):
        for dev in TensorTest.devices:
            a_data = [1, 2, 3, 4]
            b_data = [0, -1, -2, -3, -3, -4, -5, -6, -6, -7, -8, -9]
            y_data = [-8, -10, -12, -14]
            a = tF.raw_input([2, 2], a_data, dev)
            b = tF.raw_input(Shape([2, 2], 3), b_data, dev)
            a += b
            self.assertEqual(y_data, a.to_list())

    def test_TensorTest_CheckInplaceAddN1(self):
        for dev in TensorTest.devices:
            a_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            b_data = [0, -1, -2, -3]
            y_data = [1, 1, 1, 1, 5, 5, 5, 5, 9, 9, 9, 9]
            a = tF.raw_input(Shape([2, 2], 3), a_data, dev)
            b = tF.raw_input([2, 2], b_data, dev)
            a += b
            self.assertEqual(y_data, a.to_list())

    def test_TensorTest_CheckInplaceAdd(self):
        for dev in TensorTest.devices:
            a_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            b_data = [0, -1, -2, -3, -3, -4, -5, -6, -6, -7, -8, -9]
            y_data = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
            a = tF.raw_input(Shape([2, 2], 3), a_data, dev)
            b = tF.raw_input(Shape([2, 2], 3), b_data, dev)

            copied = Tensor(a)
            ref_a = a

            a += b
            self.assertEqual(y_data, a.to_list())

            self.assertEqual(y_data, a.to_list());
            self.assertEqual(a_data, copied.to_list());
            self.assertIs(ref_a, a)

    def test_TensorTest_CheckInplaceSubtractNN(self):
        for dev in TensorTest.devices:
            a_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            b_data = [0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9]
            y_data = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
            a = tF.raw_input(Shape([2, 2], 3), a_data, dev)
            b = tF.raw_input(Shape([2, 2], 3), b_data, dev)
            a -= b
            self.assertEqual(y_data, a.to_list())

    def test_TensorTest_CheckInplaceSubtract1N(self):
        for dev in TensorTest.devices:
            a_data = [1, 2, 3, 4]
            b_data = [0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9]
            y_data = [-8, -10, -12, -14]
            a = tF.raw_input([2, 2], a_data, dev)
            b = tF.raw_input(Shape([2, 2], 3), b_data, dev)
            a -= b
            self.assertEqual(y_data, a.to_list())

    def test_TensorTest_CheckInplaceSubtractN1(self):
        for dev in TensorTest.devices:
            a_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            b_data = [0, 1, 2, 3]
            y_data = [1, 1, 1, 1, 5, 5, 5, 5, 9, 9, 9, 9]
            a = tF.raw_input(Shape([2, 2], 3), a_data, dev)
            b = tF.raw_input([2, 2], b_data, dev)
            a -= b
            self.assertEqual(y_data, a.to_list())

    def test_TensorTest_CheckInplaceSubtract(self):
        for dev in TensorTest.devices:
            a_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            b_data = [0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9]
            y_data = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
            a = tF.raw_input(Shape([2, 2], 3), a_data, dev)
            b = tF.raw_input(Shape([2, 2], 3), b_data, dev)

            copied = Tensor(a)
            ref_a = a

            a -= b
            self.assertEqual(y_data, a.to_list())

            self.assertEqual(y_data, a.to_list());
            self.assertEqual(a_data, copied.to_list());
            self.assertIs(ref_a, a)

    def test_TensorTest_CheckInvalidInplaceOps(self):
        for dev in TensorTest.devices:
            shapes = [
                Shape(),
                Shape([], 3),
                Shape([2, 2], 2),
            ]
            a = tF.raw_input(Shape([2, 2], 3), [0] * 12, dev)

            for shape in shapes:
                b = tF.raw_input(shape, [0] * shape.size(), dev)
                with self.assertRaises(RuntimeError):
                    a += b
                with self.assertRaises(RuntimeError):
                    a -= b

    def test_TensorTest_CheckArgMaxDims(self):
        data = [
            0, 1, 2, 6, 7, 8, 3, 4, 5, -3, -4, -5, 0, -1, -2, -6, -7, -8,
        ]
        expected = [
            [2, 2, 2, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        for dev in TensorTest.devices:
            a = tF.raw_input(Shape([3, 3], 2), data, dev)
            for i, exp in enumerate(expected):
                self.assertEqual(exp, a.argmax(i))

    def test_TensorTest_CheckArgMaxLarge(self):
        ns = [
            1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024,
            1025, 65535, 65536, 65537,
        ]
        for dev in TensorTest.devices:
            for n in ns:
                data = list(range(n))
                random.shuffle(data)
                pos = data.index(n - 1)
                a = tF.raw_input([n], data, dev)
                self.assertEqual(pos, a.argmax(0)[0])

    def test_TensorTest_CheckArgMinDims(self):
        data = [
            3, 4, 5, 0, 1, 2, 6, 7, 8, 0, -1, -2, -6, -7, -8, -3, -4, -5,
        ]
        expected = [
            [0, 0, 0, 2, 2, 2],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        for dev in TensorTest.devices:
            a = tF.raw_input(Shape([3, 3], 2), data, dev)
            for i, exp in enumerate(expected):
                self.assertEqual(exp, a.argmin(i))

    def test_TensorTest_CheckArgMinLarge(self):
        ns = [
            1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024,
            1025, 65535, 65536, 65537,
        ]
        for dev in TensorTest.devices:
            for n in ns:
                data = list(range(n))
                random.shuffle(data)
                pos = data.index(0)
                a = tF.raw_input([n], data, dev)
                self.assertEqual(pos, a.argmin(0)[0])
