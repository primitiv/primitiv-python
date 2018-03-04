import math
import random
import sys
import unittest

from primitiv import Device
from primitiv import Shape
from primitiv import Parameter
from primitiv import Tensor
from primitiv import initializers as I
from primitiv import tensor_functions as tF

import numpy as np
from . import test_utils


class TensorForwardTest(unittest.TestCase):

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

    def test_TensorForwardTest_CheckInputByVector(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        for dev in TensorForwardTest.devices:
            y = tF.raw_input(Shape([2, 2], 3), data, dev)
            self.assertEqual(Shape([2, 2], 3), y.shape())
            self.assertIs(dev, y.device())
            self.assertEqual(data, y.to_list())

    def test_TensorForwardTest_CheckInputByParameter(self):
        data = [1, 2, 3, 4]
        for dev in TensorForwardTest.devices:
            param = Parameter([2, 2], I.Constant(0), dev)
            param.value += tF.raw_input([2, 2], data, dev)
            y = tF.parameter(param)
            self.assertEqual(Shape([2, 2]), y.shape())
            self.assertIs(dev, y.device())
            self.assertEqual(data, y.to_list())

    def test_TensorForwardTest_CheckInputByNdArray(self):
        y_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        x_data = [
            np.array([[1, 3], [2, 4]]),
            np.array([[5, 7], [6, 8]]),
            np.array([[9, 11], [10, 12]]),
        ]
        for dev in TensorForwardTest.devices:
            y = tF.input(x_data, dev)
            self.assertEqual(Shape([2, 2], 3), y.shape())
            self.assertIs(dev, y.device())
            self.assertEqual(y_data, y.to_list())

    def test_TensorForwardTest_CheckCopy(self):
        i = 0
        for dev in TensorForwardTest.devices:
            for dev2 in TensorForwardTest.devices:
                data = list(range(i, i + 12))
                print(data)
                i += 12
                x = tF.raw_input(Shape([2, 2], 3), data, dev)
                y = tF.copy(x, dev2)
                self.assertEqual(Shape([2, 2], 3), y.shape())
                self.assertIs(dev, x.device())
                self.assertIs(dev2, y.device())
                self.assertEqual(x.to_list(), y.to_list())
                y *= 2
                self.assertNotEqual(x.to_list(), y.to_list())

    def test_TensorForwardTest_CheckInvalidCopy(self):
        for dev in TensorForwardTest.devices:
            with self.assertRaises(RuntimeError):
                tF.copy(Tensor(), dev)

    def test_TensorForwardTest_CheckIdentity(self):
        test_cases = [
            (1, Shape(), [1]),
            (2, Shape([2, 2]), [1, 0, 0, 1]),
            (3, Shape([3, 3]), [1, 0, 0, 0, 1, 0, 0, 0, 1]),
            (4, Shape([4, 4]), [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
        ]
        for dev in TensorForwardTest.devices:
            Device.set_default(dev)
            for tc in test_cases:
                y = tF.identity(tc[0])
                self.assertEqual(tc[1], y.shape())
                self.assertEqual(tc[2], y.to_list())

    def test_TensorForwardTest_CheckInvalidIdentity(self):
        for dev in TensorForwardTest.devices:
            Device.set_default(dev)
            with self.assertRaises(RuntimeError):
                tF.identity(0)

    def test_TensorForwardTest_CheckPickNN(self):
        test_cases = [
            (Shape([2, 2, 2], 3), 0, [0, 0, 0],
                Shape([1, 2, 2], 3),
                [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]),
            (Shape([2, 2, 2], 3), 0, [1, 0, 1],
                Shape([1, 2, 2], 3),
                [1, 3, 5, 7, 8, 10, 12, 14, 17, 19, 21, 23]),
            (Shape([2, 2, 2], 3), 0, [0],
                Shape([1, 2, 2], 3),
                [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]),
            (Shape([2, 2, 2]), 0, [0, 1, 0],
                Shape([1, 2, 2], 3),
                [0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6]),
            (Shape([2, 2, 2], 3), 1, [0, 0, 0],
                Shape([2, 1, 2], 3),
                [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21]),
            (Shape([2, 2, 2], 3), 2, [0, 0, 0],
                Shape([2, 2, 1], 3),
                [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19]),
        ]
        for dev in TensorForwardTest.devices:
            for tc in test_cases:
                print("x_shape =", tc[0],
                      ", dim =", tc[1], ", ids = [", file=sys.stderr)
                print(tc[2], file=sys.stderr)
                print("]", file=sys.stderr)
                x_data = list(range(tc[0].size()))
                x = tF.raw_input(tc[0], x_data, dev)
                y = tF.pick(x, tc[2], tc[1])
                self.assertEqual(tc[3], y.shape())
                self.assertEqual(tc[4], y.to_list())

    def test_TensorForwardTest_CheckInvalidPick(self):
        test_cases = [
            (0, []),
            (0, [2]),
            (0, [0, 1]),
            (0, [0, 1, 2]),
            (1, [2]),
            (2, [2]),
            (3, [1]),
        ]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 2, 2], 3), [0] * 24, dev)
            for tc in test_cases:
                with self.assertRaises(RuntimeError):
                    tF.pick(x, tc[1], tc[0])

    def test_TensorForwardTest_CheckSlice(self):
        x_data = list(range(3 * 3 * 2 * 4))
        test_cases = [
            (0, 0, 1, Shape([1, 3, 2], 4),
                [0, 3, 6, 9, 12, 15,
                 18, 21, 24, 27, 30, 33,
                 36, 39, 42, 45, 48, 51,
                 54, 57, 60, 63, 66, 69]),
            (1, 0, 1, Shape([3, 1, 2], 4),
                [0, 1, 2, 9, 10, 11,
                 18, 19, 20, 27, 28, 29,
                 36, 37, 38, 45, 46, 47,
                 54, 55, 56, 63, 64, 65]),
            (2, 0, 1, Shape([3, 3, 1], 4),
                [0, 1, 2, 3, 4, 5, 6, 7, 8,
                 18, 19, 20, 21, 22, 23, 24, 25, 26,
                 36, 37, 38, 39, 40, 41, 42, 43, 44,
                 54, 55, 56, 57, 58, 59, 60, 61, 62]),
            (0, 1, 2, Shape([1, 3, 2], 4),
                [1, 4, 7, 10, 13, 16,
                 19, 22, 25, 28, 31, 34,
                 37, 40, 43, 46, 49, 52,
                 55, 58, 61, 64, 67, 70]),
            (1, 1, 2, Shape([3, 1, 2], 4),
                [3, 4, 5, 12, 13, 14,
                 21, 22, 23, 30, 31, 32,
                 39, 40, 41, 48, 49, 50,
                 57, 58, 59, 66, 67, 68]),
            (2, 1, 2, Shape([3, 3, 1], 4),
                [9, 10, 11, 12, 13, 14, 15, 16, 17,
                 27, 28, 29, 30, 31, 32, 33, 34, 35,
                 45, 46, 47, 48, 49, 50, 51, 52, 53,
                 63, 64, 65, 66, 67, 68, 69, 70, 71]),
            (0, 2, 3, Shape([1, 3, 2], 4),
                [2, 5, 8, 11, 14, 17,
                 20, 23, 26, 29, 32, 35,
                 38, 41, 44, 47, 50, 53,
                 56, 59, 62, 65, 68, 71]),
            (1, 2, 3, Shape([3, 1, 2], 4),
                [6, 7, 8, 15, 16, 17,
                 24, 25, 26, 33, 34, 35,
                 42, 43, 44, 51, 52, 53,
                 60, 61, 62, 69, 70, 71]),
            (3, 0, 1, Shape([3, 3, 2], 4), x_data),
        ]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([3, 3, 2], 4), x_data, dev)
            for tc in test_cases:
                print("dim =", tc[0], ", lower =", tc[1],
                      ", upper =", tc[2], file=sys.stderr)
                y = tF.slice(x, tc[0], tc[1], tc[2])
                self.assertEqual(tc[3], y.shape())
                self.assertEqual(tc[4], y.to_list())

    def test_TensorForwardTest_CheckInvalidSlice(self):
        test_cases = [
            (0, 0, 0), (0, 1, 0), (0, 0, 4), (0, 3, 4),
            (1, 0, 0), (1, 1, 0), (1, 0, 4), (1, 3, 4),
            (2, 0, 0), (2, 1, 0), (2, 0, 2), (2, 1, 2),
        ]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input([3, 3], [3] * 9)
            for tc in test_cases:
                with self.assertRaises(RuntimeError):
                    tF.slice(x, tc[0], tc[1], tc[2])

    def test_TensorForwardTest_CheckConcatN_3x3(self):
        y_data = [
            1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6,
        ]
        for dev in TensorForwardTest.devices:
            a = tF.raw_input([1, 3], [1, 1, 1], dev)
            b = tF.raw_input([2, 3], [2, 3, 2, 3, 2, 3], dev)
            c = tF.raw_input([3, 3], [4, 5, 6, 4, 5, 6, 4, 5, 6], dev)
            y = tF.concat([a, b, c], 0)
            self.assertEqual(Shape([6, 3]), y.shape())
            self.assertEqual(y_data, y.to_list())

    def test_TensorForwardTest_CheckConcat5x4(self):
        shapes = [
            Shape([20]),
            Shape([5, 4]),
            Shape([5, 1, 4]),
        ]
        y_data = [
            1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
        ]
        for dev in TensorForwardTest.devices:
            a = tF.raw_input([5], [1, 1, 1, 1, 1], dev)
            b = tF.raw_input([5], [2, 2, 2, 2, 2], dev)
            c = tF.raw_input([5], [3, 3, 3, 3, 3], dev)
            d = tF.raw_input([5], [4, 4, 4, 4, 4], dev)
            for i in range(3):
                y = tF.concat([a, b, c, d], i)
                self.assertEqual(shapes[i], y.shape())
                self.assertEqual(y_data, y.to_list())

    def test_TensorForwardTest_CheckConcat2_2_2x2(self):
        a_data = [
            1, 2, 3, 4, 5, 6, 7, 8,
            11, 22, 33, 44, 55, 66, 77, 88,
        ]
        b_data = [
            -1, -2, -3, -4, -5, -6, -7, -8,
            -11, -22, -33, -44, -55, -66, -77, -88,
        ]
        shapes = [
            Shape([4, 2, 2], 2),
            Shape([2, 4, 2], 2),
            Shape([2, 2, 4], 2),
            Shape([2, 2, 2, 2], 2),
            Shape([2, 2, 2, 1, 2], 2),
        ]
        y_data = [
            [1, 2, -1, -2, 3, 4, -3, -4,
             5, 6, -5, -6, 7, 8, -7, -8,
             11, 22, -11, -22, 33, 44, -33, -44,
             55, 66, -55, -66, 77, 88, -77, -88],
            [1, 2, 3, 4, -1, -2, -3, -4,
             5, 6, 7, 8, -5, -6, -7, -8,
             11, 22, 33, 44, -11, -22, -33, -44,
             55, 66, 77, 88, -55, -66, -77, -88],
            [1, 2, 3, 4, 5, 6, 7, 8,
             -1, -2, -3, -4, -5, -6, -7, -8,
             11, 22, 33, 44, 55, 66, 77, 88,
             -11, -22, -33, -44, -55, -66, -77, -88],
        ]
        for dev in TensorForwardTest.devices:
            a = tF.raw_input(Shape([2, 2, 2], 2), a_data, dev)
            b = tF.raw_input(Shape([2, 2, 2], 2), b_data, dev)
            for i in range(5):
                y = tF.concat([a, b], i)
                self.assertEqual(shapes[i], y.shape())
                self.assertEqual(y_data[i if i < 2 else 2], y.to_list())

    def test_TensorForwardTest_CheckConcatBatchBroadcast(self):
        for dev in TensorForwardTest.devices:
            y_data = [
                1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
                11, 11, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
            ]
            a = tF.raw_input(Shape([2, 1], 2), [1, 1, 11, 11], dev)
            b = tF.raw_input([2, 2], [2, 2, 2, 2], dev)
            c = tF.raw_input([2, 3], [3, 3, 3, 3, 3, 3], dev)
            y = tF.concat([a, b, c], 1)
            self.assertEqual(Shape([2, 6], 2), y.shape())
            self.assertEqual(y_data, y.to_list())

            y_data = [
                1, 1, 1, 2, 2, 3, 1, 1, 1, 2, 2, 3,
                1, 1, 1, 22, 22, 33, 1, 1, 1, 22, 22, 33,
            ]
            a = tF.raw_input([3, 2], [1, 1, 1, 1, 1, 1], dev)
            b = tF.raw_input(Shape([2, 2], 2),
                             [2, 2, 2, 2, 22, 22, 22, 22], dev)
            c = tF.raw_input(Shape([1, 2], 2), [3, 3, 33, 33], dev)
            y = tF.concat([a, b, c], 0)
            self.assertEqual(Shape([6, 2], 2), y.shape())
            self.assertEqual(y_data, y.to_list())

            y_data = [1, 2, 3, 1, 2, 33, 1, 2, 333]
            a = tF.raw_input([], [1], dev)
            b = tF.raw_input([], [2], dev)
            c = tF.raw_input(Shape([], 3), [3, 33, 333], dev)
            y = tF.concat([a, b, c], 0)
            self.assertEqual(Shape([3], 3), y.shape())
            self.assertEqual(y_data, y.to_list())

    def test_TensorForwardTest_CheckInvalidConcat(self):
        for dev in TensorForwardTest.devices:
            a = tF.zeros(Shape([1, 42], 2), dev)
            b = tF.zeros(Shape([2, 42], 2), dev)
            c = tF.zeros(Shape([1, 42], 3), dev)
            d = tF.zeros([2, 42], dev)

            tF.concat([a, b], 0)
            with self.assertRaises(RuntimeError):
                tF.concat([a, b], 1)
            with self.assertRaises(RuntimeError):
                tF.concat([a, b], 2)
            with self.assertRaises(RuntimeError):
                tF.concat([a, c], 0)
            with self.assertRaises(RuntimeError):
                tF.concat([a, c], 1)
            with self.assertRaises(RuntimeError):
                tF.concat([a, c], 2)
            with self.assertRaises(RuntimeError):
                tF.concat([b, c], 0)
            with self.assertRaises(RuntimeError):
                tF.concat([b, c], 1)
            with self.assertRaises(RuntimeError):
                tF.concat([b, c], 2)
            tF.concat([a, d], 0)
            with self.assertRaises(RuntimeError):
                tF.concat([a, d], 1)
            with self.assertRaises(RuntimeError):
                tF.concat([a, d], 2)

    def test_TensorForwardTest_CheckReshape(self):
        shapes = [
            Shape([6]), Shape([1, 6]), Shape([1, 1, 6]), Shape([1, 1, 1, 6]),
            Shape([2, 3]), Shape([2, 1, 3]), Shape([1, 2, 3]),
            Shape([2, 1, 1, 3]), Shape([1, 2, 1, 3]), Shape([1, 1, 2, 3]),
            Shape([3, 2]), Shape([3, 1, 2]), Shape([1, 3, 2]),
            Shape([3, 1, 1, 2]), Shape([1, 3, 1, 2]), Shape([1, 1, 3, 2]),
        ]
        for dev in TensorForwardTest.devices:
            data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            a = tF.raw_input(Shape([6], 2), data, dev)
            for shape in shapes:
                y1 = tF.reshape(a, shape)
                self.assertEqual(shape.resize_batch(2), y1.shape())
                self.assertEqual(data, y1.to_list())
                y2 = tF.reshape(a, shape.resize_batch(2))
                self.assertEqual(shape.resize_batch(2), y2.shape())
                self.assertEqual(data, y2.to_list())

    def test_TensorForwardTest_CheckInvalidReshape(self):
        for dev in TensorForwardTest.devices:
            a = tF.zeros(Shape([6], 2), dev)
            with self.assertRaises(RuntimeError):
                tF.reshape(a, [7])
            with self.assertRaises(RuntimeError):
                tF.reshape(a, Shape([6], 3))
            with self.assertRaises(RuntimeError):
                tF.reshape(a, Shape([7], 3))

    def test_TensorForwardTest_CheckFlatten(self):
        shapes = [
            Shape([6]), Shape([1, 6]), Shape([1, 1, 6]), Shape([1, 1, 1, 6]),
            Shape([2, 3]), Shape([2, 1, 3]), Shape([1, 2, 3]),
            Shape([2, 1, 1, 3]), Shape([1, 2, 1, 3]), Shape([1, 1, 2, 3]),
            Shape([3, 2]), Shape([3, 1, 2]), Shape([1, 3, 2]),
            Shape([3, 1, 1, 2]), Shape([1, 3, 1, 2]), Shape([1, 1, 3, 2]),
        ]
        for dev in TensorForwardTest.devices:
            data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            for shape in shapes:
                a = tF.raw_input(shape.resize_batch(2), data, dev)
                y = tF.flatten(a)
                self.assertEqual(Shape([6], 2), y.shape())
                self.assertEqual(data, y.to_list())

    def test_TensorForwardTest_CheckDuplicate(self):
        x_data = [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 2], 2), x_data, dev)
            y = +x
            self.assertEqual(Shape([2, 2], 2), y.shape())
            self.assertTrue(np.isclose(x_data, y.to_list()).all())

    def test_TensorForwardTest_CheckNegate(self):
        x_data = [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
        y_data = [-1000, -100, -10, -1, -0.1, -0.01, -0.001, -0.0001]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 2], 2), x_data, dev)
            y = -x
            self.assertEqual(Shape([2, 2], 2), y.shape())
            self.assertTrue(np.isclose(y_data, y.to_list()).all())

    def test_TensorForwardTest_CheckAddConst(self):
        x_data = [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
        k = 1
        y_data = [1001, 101, 11, 2, 1.1, 1.01, 1.001, 1.0001]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 2], 2), x_data, dev)
            y1 = k + x
            self.assertEqual(Shape([2, 2], 2), y1.shape())
            self.assertTrue(np.isclose(y_data, y1.to_list()).all())
            y2 = x + k
            self.assertEqual(Shape([2, 2], 2), y2.shape())
            self.assertTrue(np.isclose(y_data, y2.to_list()).all())

    def test_TensorForwardTest_CheckAddScalar(self):
        x_data = [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
        k_data = [10, 1]
        y_data = [1010, 110, 20, 11, 1.1, 1.01, 1.001, 1.0001]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 2], 2), x_data, dev)
            k = tF.raw_input(Shape([], 2), k_data, dev)
            y1 = k + x
            self.assertEqual(Shape([2, 2], 2), y1.shape())
            self.assertTrue(np.isclose(y_data, y1.to_list()).all())
            y2 = x + k
            self.assertEqual(Shape([2, 2], 2), y2.shape())
            self.assertTrue(np.isclose(y_data, y2.to_list()).all())

    def test_TensorForwardTest_CheckAddScalarBatchBroadcast(self):
        x_data = [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
        k_data = [1]
        y_data = [1001, 101, 11, 2, 1.1, 1.01, 1.001, 1.0001]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 2], 2), x_data, dev)
            k = tF.raw_input([], k_data, dev)
            y1 = k + x
            self.assertEqual(Shape([2, 2], 2), y1.shape())
            self.assertTrue(np.isclose(y_data, y1.to_list()).all())
            y2 = x + k
            self.assertEqual(Shape([2, 2], 2), y2.shape())
            self.assertTrue(np.isclose(y_data, y2.to_list()).all())
        x_data = [1000, 100, 10, 1]
        k_data = [10, 1]
        y_data = [1010, 110, 20, 11, 1001, 101, 11, 2]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input([2, 2], x_data, dev)
            k = tF.raw_input(Shape([], 2), k_data, dev)
            y1 = k + x
            self.assertEqual(Shape([2, 2], 2), y1.shape())
            self.assertEqual(y_data, y1.to_list())
            y2 = x + k
            self.assertEqual(Shape([2, 2], 2), y2.shape())
            self.assertEqual(y_data, y2.to_list())

    def test_TensorForwardTest_CheckAdd(self):
        a_data = [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
        b_data = [   0, 100, 20, 3, 0.4, 0.05, 0.006, 0.0007]
        y_data = [1000, 200, 30, 4, 0.5, 0.06, 0.007, 0.0008]
        for dev in TensorForwardTest.devices:
            a = tF.raw_input(Shape([2, 2], 2), a_data, dev)
            b = tF.raw_input(Shape([2, 2], 2), b_data, dev)
            y1 = a + b
            self.assertEqual(Shape([2, 2], 2), y1.shape())
            self.assertTrue(np.isclose(y_data, y1.to_list()).all())
            y2 = b + a
            self.assertEqual(Shape([2, 2], 2), y2.shape())
            self.assertTrue(np.isclose(y_data, y2.to_list()).all())

    def test_TensorForwardTest_CheckAddBatchBroadcast(self):
        a_data = [0, 1, 2, 3]
        b_data = [0, 0, 0, 0, 4, 4, 4, 4]
        y_data = [0, 1, 2, 3, 4, 5, 6, 7]
        for dev in TensorForwardTest.devices:
            a = tF.raw_input([2, 2], a_data, dev)
            b = tF.raw_input(Shape([2, 2], 2), b_data, dev)
            y1 = a + b
            self.assertEqual(Shape([2, 2], 2), y1.shape())
            self.assertEqual(y_data, y1.to_list())
            y2 = b + a
            self.assertEqual(Shape([2, 2], 2), y2.shape())
            self.assertEqual(y_data, y2.to_list())

    def test_TensorForwardTest_CheckSubtractConst(self):
        x_data = [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
        k = 1
        y1_data = [-999, -99, -9, 0, 0.9, 0.99, 0.999, 0.9999]
        y2_data = [999, 99, 9, 0, -0.9, -0.99, -0.999, -0.9999]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 2], 2), x_data, dev)
            y1 = k - x
            self.assertEqual(Shape([2, 2], 2), y1.shape())
            self.assertTrue(np.isclose(y1_data, y1.to_list()).all())
            y2 = x - k
            self.assertEqual(Shape([2, 2], 2), y2.shape())
            self.assertTrue(np.isclose(y2_data, y2.to_list()).all())

    def test_TensorForwardTest_CheckSubtractScalar(self):
        x_data = [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
        k_data = [10, 1]
        y1_data = [-990, -90, 0, 9, 0.9, 0.99, 0.999, 0.9999]
        y2_data = [990, 90, 0, -9, -0.9, -0.99, -0.999, -0.9999]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 2], 2), x_data, dev)
            k = tF.raw_input(Shape([], 2), k_data, dev)
            y1 = k - x
            self.assertEqual(Shape([2, 2], 2), y1.shape())
            self.assertTrue(np.isclose(y1_data, y1.to_list()).all())
            y2 = x - k
            self.assertEqual(Shape([2, 2], 2), y2.shape())
            self.assertTrue(np.isclose(y2_data, y2.to_list()).all())

    def test_TensorForwardTest_CheckSubtractScalarBatchBroadcast(self):
        x_data = [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
        k_data = [1]
        y1_data = [-999, -99, -9, 0, 0.9, 0.99, 0.999, 0.9999]
        y2_data = [999, 99, 9, 0, -0.9, -0.99, -0.999, -0.9999]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 2], 2), x_data, dev)
            k = tF.raw_input([], k_data, dev)
            y1 = k - x
            self.assertEqual(Shape([2, 2], 2), y1.shape())
            self.assertTrue(np.isclose(y1_data, y1.to_list()).all())
            y2 = x - k
            self.assertEqual(Shape([2, 2], 2), y2.shape())
            self.assertTrue(np.isclose(y2_data, y2.to_list()).all())
        x_data = [1000, 100, 10, 1]
        k_data = [10, 1]
        y1_data = [-990, -90, 0, 9, -999, -99, -9, 0]
        y2_data = [990, 90, 0, -9, 999, 99, 9, 0]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input([2, 2], x_data, dev)
            k = tF.raw_input(Shape([], 2), k_data, dev)
            y1 = k - x
            self.assertEqual(Shape([2, 2], 2), y1.shape())
            self.assertEqual(y1_data, y1.to_list())
            y2 = x - k
            self.assertEqual(Shape([2, 2], 2), y2.shape())
            self.assertEqual(y2_data, y2.to_list())

    def test_TensorForwardTest_CheckSubtract(self):
        a_data = [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
        b_data = [   0, 100, 20, 3, 0.4, 0.05, 0.006, 0.0007]
        y1_data = [1000, 0, -10, -2, -0.3, -0.04, -0.005, -0.0006]
        y2_data = [-1000, 0, 10, 2, 0.3, 0.04, 0.005, 0.0006]
        for dev in TensorForwardTest.devices:
            a = tF.raw_input(Shape([2, 2], 2), a_data, dev)
            b = tF.raw_input(Shape([2, 2], 2), b_data, dev)
            y1 = a - b
            self.assertEqual(Shape([2, 2], 2), y1.shape())
            self.assertTrue(np.isclose(y1_data, y1.to_list()).all())
            y2 = b - a
            self.assertEqual(Shape([2, 2], 2), y2.shape())
            self.assertTrue(np.isclose(y2_data, y2.to_list()).all())

    def test_TensorForwardTest_CheckSubtractBatchBroadcast(self):
        a_data = [0, 1, 2, 3]
        b_data = [0, 0, 0, 0, 4, 4, 4, 4]
        y1_data = [0, 1, 2, 3, -4, -3, -2, -1]
        y2_data = [0, -1, -2, -3, 4, 3, 2, 1]
        for dev in TensorForwardTest.devices:
            a = tF.raw_input([2, 2], a_data, dev)
            b = tF.raw_input(Shape([2, 2], 2), b_data, dev)
            y1 = a - b
            self.assertEqual(Shape([2, 2], 2), y1.shape())
            self.assertTrue(y1_data, y1.to_list())
            y2 = b - a
            self.assertEqual(Shape([2, 2], 2), y2.shape())
            self.assertTrue(y2_data, y2.to_list())

    def test_TensorForwardTest_CheckMultiplyConst(self):
        x_data = [1000, -100, 10, -1, 0.1, -0.01, 0.001, -0.0001]
        k = 10
        y_data = [10000, -1000, 100, -10, 1, -0.1, 0.01, -0.001]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 2], 2), x_data, dev)
            y1 = k * x
            self.assertEqual(Shape([2, 2], 2), y1.shape())
            self.assertTrue(np.isclose(y_data, y1.to_list()).all())
            y2 = x * k
            self.assertEqual(Shape([2, 2], 2), y2.shape())
            self.assertTrue(np.isclose(y_data, y2.to_list()).all())

    def test_TensorForwardTest_CheckMultiplyScalar(self):
        x_data = [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
        k_data = [0.1, 10]
        y_data = [100, 10, 1, 0.1, 1, 0.1, 0.01, 0.001]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 2], 2), x_data, dev)
            k = tF.raw_input(Shape([], 2), k_data, dev)
            y1 = k * x
            self.assertEqual(Shape([2, 2], 2), y1.shape())
            self.assertTrue(np.isclose(y_data, y1.to_list()).all())
            y2 = x * k
            self.assertEqual(Shape([2, 2], 2), y2.shape())
            self.assertTrue(np.isclose(y_data, y2.to_list()).all())

    def test_TensorForwardTest_CheckMultiplyScalarBatchBroadcast(self):
        x_data = [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
        k_data = [10]
        y_data = [10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 2], 2), x_data, dev)
            k = tF.raw_input([], k_data, dev)
            y1 = k * x
            self.assertEqual(Shape([2, 2], 2), y1.shape())
            self.assertTrue(np.isclose(y_data, y1.to_list()).all())
            y2 = x * k
            self.assertEqual(Shape([2, 2], 2), y2.shape())
            self.assertTrue(np.isclose(y_data, y2.to_list()).all())
        x_data = [1000, 100, 10, 1]
        k_data = [0.1, 10]
        y_data = [100, 10, 1, 0.1, 10000, 1000, 100, 10]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input([2, 2], x_data, dev)
            k = tF.raw_input(Shape([], 2), k_data, dev)
            y1 = k * x
            self.assertEqual(Shape([2, 2], 2), y1.shape())
            self.assertTrue(np.isclose(y_data, y1.to_list()).all())
            y2 = x * k
            self.assertEqual(Shape([2, 2], 2), y2.shape())
            self.assertTrue(np.isclose(y_data, y2.to_list()).all())

    def test_TensorForwardTest_CheckMultiply(self):
        a_data = [1000, -100, 10, -1, 0.1, -0.01, 0.001, -0.0001]
        b_data = [0, 1, 2, 3, -4, -5, -6, -7]
        y_data = [0, -100, 20, -3, -0.4, 0.05, -0.006, 0.0007]
        for dev in TensorForwardTest.devices:
            a = tF.raw_input(Shape([2, 2], 2), a_data, dev)
            b = tF.raw_input(Shape([2, 2], 2), b_data, dev)
            y1 = a * b
            self.assertEqual(Shape([2, 2], 2), y1.shape())
            self.assertTrue(np.isclose(y_data, y1.to_list()).all())
            y2 = b * a
            self.assertEqual(Shape([2, 2], 2), y2.shape())
            self.assertTrue(np.isclose(y_data, y2.to_list()).all())

    def test_TensorForwardTest_CheckMultiplyBatchBroadcast(self):
        a_data = [0, 1, 2, 3]
        b_data = [1, 1, 1, 1, 0, 1, 2, 3]
        y_data = [0, 1, 2, 3, 0, 1, 4, 9]
        for dev in TensorForwardTest.devices:
            a = tF.raw_input([2, 2], a_data, dev)
            b = tF.raw_input(Shape([2, 2], 2), b_data, dev)
            y1 = a * b
            self.assertEqual(Shape([2, 2], 2), y1.shape())
            self.assertEqual(y_data, y1.to_list())
            y2 = b * a
            self.assertEqual(Shape([2, 2], 2), y2.shape())
            self.assertEqual(y_data, y2.to_list())

    def test_TensorForwardTest_CheckDivideConst(self):
        x_data = [1000, -100, 10, -1, 0.1, -0.01, 0.001, -0.0001]
        k = 10
        y1_data = [0.01, -0.1, 1, -10, 100, -1000, 10000, -100000]
        y2_data = [
            100, -10, 1, -0.1, 0.01, -0.001, 0.0001, -0.00001,
        ]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 2], 2), x_data, dev)
            y1 = k / x
            self.assertEqual(Shape([2, 2], 2), y1.shape())
            self.assertTrue(np.isclose(y1_data, y1.to_list()).all())
            y2 = x / k
            self.assertEqual(Shape([2, 2], 2), y2.shape())
            self.assertTrue(np.isclose(y2_data, y2.to_list()).all())

    def test_TensorForwardTest_CheckDivideScalar(self):
      x_data = [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
      k_data = [10, 0.1]
      y1_data = [0.01, 0.1, 1, 10, 1, 10, 100, 1000]
      y2_data = [100, 10, 1, 0.1, 1, 0.1, 0.01, 0.001]
      for dev in TensorForwardTest.devices:
          x = tF.raw_input(Shape([2, 2], 2), x_data, dev)
          k = tF.raw_input(Shape([], 2), k_data, dev)
          y1 = k / x
          self.assertEqual(Shape([2, 2], 2), y1.shape())
          self.assertTrue(np.isclose(y1_data, y1.to_list()).all())
          y2 = x / k
          self.assertEqual(Shape([2, 2], 2), y2.shape())
          self.assertTrue(np.isclose(y2_data, y2.to_list()).all())

    def test_TensorForwardTest_CheckDivideScalarBatchBroadcast(self):
        x_data = [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
        k_data = [10]
        y1_data = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
        y2_data = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 2], 2), x_data, dev)
            k = tF.raw_input([], k_data, dev)
            y1 = k / x
            self.assertEqual(Shape([2, 2], 2), y1.shape())
            self.assertTrue(np.isclose(y1_data, y1.to_list()).all())
            y2 = x / k
            self.assertEqual(Shape([2, 2], 2), y2.shape())
            self.assertTrue(np.isclose(y2_data, y2.to_list()).all())
        x_data = [1000, 100, 10, 1]
        k_data = [10, 0.1]
        y1_data = [0.01, 0.1, 1, 10, 0.0001, 0.001, 0.01, 0.1]
        y2_data = [100, 10, 1, 0.1, 10000, 1000, 100, 10]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input([2, 2], x_data, dev)
            k = tF.raw_input(Shape([], 2), k_data, dev)
            y1 = k / x
            self.assertEqual(Shape([2, 2], 2), y1.shape())
            self.assertTrue(np.isclose(y1_data, y1.to_list()).all())
            y2 = x / k
            self.assertEqual(Shape([2, 2], 2), y2.shape())
            self.assertTrue(np.isclose(y2_data, y2.to_list()).all())

    def test_TensorForwardTest_CheckDivide(self):
        a_data = [1000, -100, 10, -1, 0.1, -0.01, 0.001, -0.0001]
        b_data = [1, 2, 3, 4, -5, -6, -7, -8]
        y1_data = [
            1000, -50, 10.0/3, -0.25, -0.02, 0.01/6, -0.001/7, 1.25e-5,
        ]
        y2_data = [0.001, -0.02, 0.3, -4, -50, 600, -7000, 80000]
        for dev in TensorForwardTest.devices:
            a = tF.raw_input(Shape([2, 2], 2), a_data, dev)
            b = tF.raw_input(Shape([2, 2], 2), b_data, dev)
            y1 = a / b
            self.assertEqual(Shape([2, 2], 2), y1.shape())
            self.assertTrue(np.isclose(y1_data, y1.to_list()).all())
            y2 = b / a
            self.assertEqual(Shape([2, 2], 2), y2.shape())
            self.assertTrue(np.isclose(y2_data, y2.to_list()).all())

    def test_TensorForwardTest_CheckDivideBatchBroadcast(self):
        a_data = [1, 2, 3, 4]
        b_data = [1, 1, 1, 1, 1, 2, 3, 4]
        y1_data = [1, 2, 3, 4, 1, 1, 1, 1]
        y2_data = [1, 0.5, 1.0/3, 0.25, 1, 1, 1, 1]
        for dev in TensorForwardTest.devices:
            a = tF.raw_input([2, 2], a_data, dev)
            b = tF.raw_input(Shape([2, 2], 2), b_data, dev)
            y1 = a / b
            self.assertEqual(Shape([2, 2], 2), y1.shape())
            self.assertTrue(np.isclose(y1_data, y1.to_list()).all())
            y2 = b / a
            self.assertEqual(Shape([2, 2], 2), y2.shape())
            self.assertTrue(np.isclose(y2_data, y2.to_list()).all())

    def test_TensorForwardTest_CheckInvalidArithmeticOps(self):
        sa = [
            Shape([2, 2], 2), Shape([2, 2], 2), Shape([2, 2], 2),
        ]
        sb = [
            Shape([2, 2], 3), Shape([3, 3], 2), Shape([3, 3], 3),
        ]
        for dev in TensorForwardTest.devices:
            for ssa, ssb in zip(sa, sb):
                a = tF.zeros(ssa, dev)
                b = tF.zeros(ssb, dev)
                with self.assertRaises(RuntimeError):
                    a + b
                with self.assertRaises(RuntimeError):
                    a - b
                with self.assertRaises(RuntimeError):
                    a * b
                with self.assertRaises(RuntimeError):
                    a / b

    def test_TensorForwardTest_CheckTranspose11(self):
        for dev in TensorForwardTest.devices:
            x_data = [42]
            y_data = [42]
            x = tF.raw_input([], x_data, dev)
            y = tF.transpose(x)
            self.assertEqual(Shape(), y.shape())
            self.assertEqual(y_data, y.to_list())

    def test_TensorForwardTest_CheckTransposeN1(self):
        x_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        y_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input([12], x_data, dev)
            y = tF.transpose(x)
            self.assertEqual(Shape([1, 12]), y.shape())
            self.assertEqual(y_data, y.to_list())

    def test_TensorForwardTest_CheckTranspose1N(self):
        x_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        y_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([1, 3], 4), x_data, dev)
            y = tF.transpose(x)
            self.assertEqual(Shape([3], 4), y.shape())
            self.assertEqual(y_data, y.to_list())

    def test_TensorForwardTest_CheckTransposeNN(self):
        x_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        y_data = [1, 3, 2, 4, 5, 7, 6, 8, 9, 11, 10, 12]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 2], 3), x_data, dev)
            y = tF.transpose(x)
            self.assertEqual(Shape([2, 2], 3), y.shape())
            self.assertEqual(y_data, y.to_list())

    def test_TensorForwardTest_CheckTransposeMN(self):
        x_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        y_data = [1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 3], 2), x_data, dev)
            y = tF.transpose(x)
            self.assertEqual(Shape([3, 2], 2), y.shape())
            self.assertEqual(y_data, y.to_list())

    def test_TensorForwardTest_CheckInvalidTranspose(self):
        for dev in TensorForwardTest.devices:
            x = tF.zeros([2, 3, 4], dev)
            with self.assertRaises(RuntimeError):
                tF.transpose(x)

    def test_TensorForwardTest_CheckMatMulAA(self):
        x_data = [1, 2, 3, 4, 1, 0, 0, 1, 0, 2, 3, 0]
        y_data = [7, 10, 15, 22, 1, 0, 0, 1, 6, 0, 0, 6]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 2], 3), x_data, dev)
            y1 = tF.matmul(x, x)
            y2 = x @ x
            self.assertEqual(Shape([2, 2], 3), y1.shape())
            self.assertEqual(y_data, y1.to_list())
            self.assertEqual(Shape([2, 2], 3), y2.shape())
            self.assertEqual(y_data, y2.to_list())

    def test_TensorForwardTest_CheckMatMulAB(self):
        a_data = [
            1, 1000, 1,
            10, 100, 100,
            100, 10, 10000,
            1000, 1, 1000000,
        ]
        b_data = [
            0, 2, 4, 6,
            1, 3, 5, 7,
            8, 6, 4, 2,
            9, 7, 5, 3,
            2, 3, 5, 7,
            9, 4, 1, 0,
        ]
        y_data = [
            6420, 246, 6040200,
            7531, 1357, 7050301,
            2468, 8642, 2040608,
            3579, 9753, 3050709,
            7532, 2357, 7050302,
            149, 9410, 10409,
        ]
        for dev in TensorForwardTest.devices:
            a = tF.raw_input([3, 4], a_data, dev)
            b = tF.raw_input([4, 6], b_data, dev)
            y1 = tF.matmul(a, b)
            y2 = a @ b
            self.assertEqual(Shape([3, 6]), y1.shape())
            self.assertEqual(y_data, y1.to_list())
            self.assertEqual(Shape([3, 6]), y2.shape())
            self.assertEqual(y_data, y2.to_list())

    def test_TensorForwardTest_CheckMatMulBatchBroadcast1N(self):
        a_data = [10, 1000, 1, 100]
        b_data = [1, 2, 3, 4, 5, 6, 7, 8]
        y_data = [12, 1200, 34, 3400, 56, 5600, 78, 7800]
        for dev in TensorForwardTest.devices:
            a = tF.raw_input([2, 2], a_data, dev)
            b = tF.raw_input(Shape([2, 2], 2), b_data, dev)
            y = tF.matmul(a, b)
            self.assertEqual(Shape([2, 2], 2), y.shape())
            self.assertEqual(y_data, y.to_list())

    def test_TensorForwardTest_CheckMatMulBatchBroadcastN1(self):
        a_data = [1, 2, 3, 4, 5, 6, 7, 8]
        b_data = [10, 1, 1000, 100]
        y_data = [13, 24, 1300, 2400, 57, 68, 5700, 6800]
        for dev in TensorForwardTest.devices:
            a = tF.raw_input(Shape([2, 2], 2), a_data, dev)
            b = tF.raw_input([2, 2], b_data, dev)
            y = tF.matmul(a, b)
            self.assertEqual(Shape([2, 2], 2), y.shape())
            self.assertEqual(y_data, y.to_list())

    def test_TensorForwardTest_CheckMatMulLarge(self):
        N = 123
        a_data = [0] * (N * N)
        b_data = [0] * (N * N)
        y1_data = [0] * (N * N)
        y2_data = [0] * (N * N)
        k = 0
        for i in range(N):
            k += i * i
        for i in range(N):
            for j in range(N):
                a_data[i + j * N] = i
                b_data[i + j * N] = j
                y1_data[i + j * N] = N * i * j
                y2_data[i + j * N] = k
        for dev in TensorForwardTest.devices:
            a = tF.raw_input(Shape([N, N]), a_data, dev)
            b = tF.raw_input([N, N], b_data, dev)
            y1 = tF.matmul(a, b)
            y2 = tF.matmul(b, a)
            self.assertEqual(Shape([N, N]), y1.shape())
            self.assertEqual(Shape([N, N]), y2.shape())
            self.assertEqual(y1_data, y1.to_list())
            self.assertEqual(y2_data, y2.to_list())

    def test_TensorForwardTest_CheckInvalidMatMul(self):
        for dev in TensorForwardTest.devices:
            a = tF.zeros([2, 3], dev)
            b = tF.zeros([], dev)
            with self.assertRaises(RuntimeError):
                tF.matmul(a, b)
            a = tF.zeros([], dev)
            b = tF.zeros([2, 3], dev)
            with self.assertRaises(RuntimeError):
                tF.matmul(a, b)
            a = tF.zeros([2, 3, 4], dev)
            b = tF.zeros([4], dev)
            with self.assertRaises(RuntimeError):
                tF.matmul(a, b)
            a = tF.zeros([1, 2], dev)
            b = tF.zeros([2, 3, 4], dev)
            with self.assertRaises(RuntimeError):
                tF.matmul(a, b)
            a = tF.zeros([2, 3], dev)
            b = tF.zeros([2, 3], dev)
            with self.assertRaises(RuntimeError):
                tF.matmul(a, b)

    def test_TensorForwardTest_CheckSqrt(self):
        x_data = [
            0, 1, 2, 3, 4, 5,
            0, 1, 4, 9, 16, 25,
        ]
        y_data = [
            0, 1, 1.41421356, 1.73205041, 2, 2.23606798,
            0, 1, 2, 3, 4, 5,
        ]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 3], 2), x_data, dev)
            y = tF.sqrt(x)
            self.assertEqual(Shape([2, 3], 2), y.shape())
            self.assertTrue(np.isclose(y_data, y.to_list()).all())

    def test_TensorForwardTest_CheckExp(self):
        x_data = [
            0, .5, 1, 2, 4, 8,
            0, -.5, -1, -2, -4, -8,
        ]
        y_data = [
            1, 1.6487213, 2.7182818, 7.3890561, 54.598150, 2980.9580,
            1, .60653066, .36787944, .13533528, .018315639, .00033546263,
        ]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 3], 2), x_data, dev)
            y = tF.exp(x)
            self.assertEqual(Shape([2, 3], 2), y.shape())
            self.assertTrue(np.isclose(y_data, y.to_list()).all())

    def test_TensorForwardTest_CheckLog(self):
        x_data = [
            0.01, .5, 1, 2, 4, 8,
            0.01, .5, 1, 2, 4, 8,
        ]
        y_data = [
            -4.60517019, -0.69314718, 0, 0.69314718, 1.38629436, 2.07944154,
            -4.60517019, -0.69314718, 0, 0.69314718, 1.38629436, 2.07944154,
        ]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 3], 2), x_data, dev)
            y = tF.log(x)
            self.assertEqual(Shape([2, 3], 2), y.shape())
            self.assertTrue(np.isclose(y_data, y.to_list()).all())

    def test_TensorForwardTest_CheckPow(self):
        x_data = [
            0.01, .5, 1, 2, 4, 8,
            0.01, .5, 1, 2, 4, 8,
        ]
        y_data = [
            0.00001, 0.17677670, 1, 5.65685425, 32, 181.01933598,
            0.00001, 0.17677670, 1, 5.65685425, 32, 181.01933598,
        ]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 3], 2), x_data, dev)
            y1 = tF.pow(x, 2.5)
            y2 = x ** 2.5
            self.assertEqual(Shape([2, 3], 2), y1.shape())
            self.assertTrue(np.isclose(y_data, y1.to_list()).all())
            self.assertEqual(Shape([2, 3], 2), y2.shape())
            self.assertTrue(np.isclose(y_data, y2.to_list()).all())

    def test_TensorForwardTest_CheckIPowPositive(self):
        x_data = [
            0.01, .5, 1, 2, 4, 8,
            -0.01, -.5, -1, -2, -4, -8,
        ]
        y_data = [
            0.000001, 0.125, 1, 8, 64, 512,
            -0.000001, -0.125, -1, -8, -64, -512,
        ]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 3], 2), x_data, dev)
            y1 = tF.pow(x, 3)
            y2 = x ** 3
            self.assertEqual(Shape([2, 3], 2), y1.shape())
            self.assertTrue(np.isclose(y_data, y1.to_list()).all())
            self.assertEqual(Shape([2, 3], 2), y2.shape())
            self.assertTrue(np.isclose(y_data, y2.to_list()).all())

    def test_TensorForwardTest_CheckIPowNegative(self):
        x_data = [
            0.01, .5, 1, 2, 4, 8,
            -0.01, -.5, -1, -2, -4, -8,
        ]
        y_data = [
            1000000, 8, 1, 0.125, 0.015625, 0.001953125,
            -1000000, -8, -1, -0.125, -0.015625, -0.001953125,
        ]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 3], 2), x_data, dev)
            y1 = tF.pow(x, -3)
            y2 = x ** -3
            self.assertEqual(Shape([2, 3], 2), y1.shape())
            self.assertTrue(np.isclose(y_data, y1.to_list()).all())
            self.assertEqual(Shape([2, 3], 2), y2.shape())
            self.assertTrue(np.isclose(y_data, y2.to_list()).all())

    def test_TensorForwardTest_CheckIPowUpperBound(self):
        x_data = [
            1, -1, 1, -1, 1, -1,
            1, -1, 1, -1, 1, -1,
        ]
        y_data = [
            1, -1, 1, -1, 1, -1,
            1, -1, 1, -1, 1, -1,
        ]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 3], 2), x_data, dev)
            y1 = tF.pow(x, 0x7fffffff)
            y2 = x ** 0x7fffffff
            self.assertEqual(Shape([2, 3], 2), y1.shape())
            self.assertTrue(np.isclose(y_data, y1.to_list()).all())
            self.assertEqual(Shape([2, 3], 2), y2.shape())
            self.assertTrue(np.isclose(y_data, y2.to_list()).all())

    def test_TensorForwardTest_CheckIPowLowerBound(self):
        x_data = [
            1, -1, 1, -1, 1, -1,
            1, -1, 1, -1, 1, -1,
        ]
        y_data  = [
            1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1,
        ]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 3], 2), x_data, dev)
            y1 = tF.pow(x, -2147483648) # 0x80000000
            y2 = x ** -2147483648
            self.assertEqual(Shape([2, 3], 2), y1.shape())
            self.assertTrue(np.isclose(y_data, y1.to_list()).all())
            self.assertEqual(Shape([2, 3], 2), y2.shape())
            self.assertTrue(np.isclose(y_data, y2.to_list()).all())

    def test_TensorForwardTest_CheckIPowPositiveConvergence(self):
        x_data = [
            0.9999999, -0.9999999, 0.9999999, -0.9999999, 0.9999999, -0.9999999,
            0.9999999, -0.9999999, 0.9999999, -0.9999999, 0.9999999, -0.9999999,
        ]
        y_data = [
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 3], 2), x_data, dev)
            y1 = tF.pow(x, 0x7fffffff)
            y2 = x ** 0x7fffffff
            self.assertEqual(Shape([2, 3], 2), y1.shape())
            self.assertEqual(y_data, y1.to_list())
            self.assertEqual(Shape([2, 3], 2), y2.shape())
            self.assertEqual(y_data, y2.to_list())

    def test_TensorForwardTest_CheckIPowNegativeConvergence(self):
        x_data = [
            1.000001, -1.000001, 1.000001, -1.000001, 1.000001, -1.000001,
            1.000001, -1.000001, 1.000001, -1.000001, 1.000001, -1.000001,
        ]
        y_data = [
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 3], 2), x_data, dev)
            y1 = tF.pow(x, -2147483648) # 0x80000000
            y2 = x ** -2147483648
            self.assertEqual(Shape([2, 3], 2), y1.shape())
            self.assertEqual(y_data, y1.to_list())
            self.assertEqual(Shape([2, 3], 2), y2.shape())
            self.assertEqual(y_data, y2.to_list())

    def test_TensorForwardTest_CheckTanh(self):
        x_data = [
            0, .5, 1, 2, 4, 8,
            0, -.5, -1, -2, -4, -8,
        ]
        y_data = [
            0, .46211716, .76159416, .96402758, .99932930, .99999977,
            0, -.46211716, -.76159416, -.96402758, -.99932930, -.99999977,
        ]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 3], 2), x_data, dev)
            y = tF.tanh(x)
            self.assertEqual(Shape([2, 3], 2), y.shape())
            self.assertTrue(np.isclose(y_data, y.to_list()).all())

    def test_TensorForwardTest_CheckSigmoid(self):
        x_data = [
            0, .5, 1, 2, 3, 4,
            0, -.5, -1, -2, -3, -4,
        ]
        y_data = [
            .5, .62245933, .73105858, .88079708, .95257413, .98201379,
            .5, .37754067, .26894142, .11920292, .047425873, .017986210,
        ]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 3], 2), x_data, dev)
            y = tF.sigmoid(x)
            self.assertEqual(Shape([2, 3], 2), y.shape())
            self.assertTrue(np.isclose(y_data, y.to_list()).all())

    def test_TensorForwardTest_CheckSoftplus(self):
        x_data = [
            0, .5, 1, 2, 3, 4,
            0, -.5, -1, -2, -3, -4,
        ]
        y_data = [
            .69314718, .97407698, 1.3132617, 2.1269280, 3.0485874, 4.0181499,
            .69314718, .47407698, .31326169, .12692801, .048587352, .018149928,
        ]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 3], 2), x_data, dev)
            y = tF.softplus(x)
            self.assertEqual(Shape([2, 3], 2), y.shape())
            self.assertTrue(np.isclose(y_data, y.to_list(), 0, 1e-6).all())

    def test_TensorForwardTest_CheckSin(self):
        x_data = [
            0, .5, 1, 2, 3, 4,
            0, -.5, -1, -2, -3, -4,
        ]
        y_data = [
            0, .47942554, .84147098, .90929743, .14112001, -.75680250,
            0, -.47942554, -.84147098, -.90929743, -.14112001, .75680250,
        ]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 3], 2), x_data, dev)
            y = tF.sin(x)
            self.assertEqual(Shape([2, 3], 2), y.shape())
            self.assertTrue(np.isclose(y_data, y.to_list()).all())

    def test_TensorForwardTest_CheckCos(self):
        x_data = [
            0, .5, 1, 2, 3, 4,
            0, -.5, -1, -2, -3, -4,
        ]
        y_data = [
            1, .87758256, .54030231, -.41614684, -.98999250, -.65364362,
            1, .87758256, .54030231, -.41614684, -.98999250, -.65364362,
        ]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 3], 2), x_data, dev)
            y = tF.cos(x)
            self.assertEqual(Shape([2, 3], 2), y.shape())
            self.assertTrue(np.isclose(y_data, y.to_list()).all())

    def test_TensorForwardTest_CheckTan(self):
        x_data = [
            0, .5, 1, 2, 3, 4,
            0, -.5, -1, -2, -3, -4,
        ]
        y_data = [
            0, .54630249, 1.5574077, -2.1850399, -.14254654, 1.1578213,
            0, -.54630249, -1.5574077, 2.1850399, .14254654, -1.1578213,
        ]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 3], 2), x_data, dev)
            y = tF.tan(x)
            self.assertEqual(Shape([2, 3], 2), y.shape())
            self.assertTrue(np.isclose(y_data, y.to_list()).all())

    def test_TensorForwardTest_CheckReLU(self):
        x_data = [
            0, .5, 1, 2, 4, 8,
            0, -.5, -1, -2, -4, -8,
        ]
        y_data = [
            0, .5, 1, 2, 4, 8,
            0, 0, 0, 0, 0, 0,
        ]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 3], 2), x_data, dev)
            y = tF.relu(x)
            self.assertEqual(Shape([2, 3], 2), y.shape())
            self.assertEqual(y_data, y.to_list())

    def test_TensorForwardTest_CheckLReLU(self):
        x_data = [
            0, .5, 1, 2, 4, 8,
            0, -.5, -1, -2, -4, -8,
        ]
        y_data = [
            0, .5, 1, 2, 4, 8,
            0, -.005, -.01, -.02, -.04, -.08,
        ]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 3], 2), x_data, dev)
            y = tF.lrelu(x)
            self.assertEqual(Shape([2, 3], 2), y.shape())
            self.assertTrue(np.isclose(y_data, y.to_list()).all())

    def test_TensorForwardTest_CheckPReLU(self):
        ks = [.01, .1, 1., 10., 100., -.01, -.1, -1., -10., -100.]
        for dev in TensorForwardTest.devices:
            for k in ks:
                x_data = [
                    0, .5, 1, 2, 4, 8,
                    0, -.5, -1, -2, -4, -8,
                ]
                y_data = [
                    0, .5, 1, 2, 4, 8,
                    0, -.5 * k, -k, -2 * k, -4 * k, -8 * k,
                ]
                x = tF.raw_input(Shape([2, 3], 2), x_data, dev)
                y = tF.prelu(x, k)
                self.assertEqual(Shape([2, 3], 2), y.shape())
                self.assertTrue(np.isclose(y_data, y.to_list()).all())

    def test_TensorForwardTest_CheckELU(self):
        ks = [.01, .1, 1., 10., 100., -.01, -.1, -1., -10., -100.]
        for dev in TensorForwardTest.devices:
            for k in ks:
                x_data = [
                    0, .5, 1, 2, 4, 8,
                    0, -.5, -1, -2, -4, -8,
                ]
                y_data = [
                    0, .5, 1, 2, 4, 8,
                    0, -.39346934 * k, -.63212056 * k,
                    -.86466472 * k, -.98168436 * k, -.99966454 * k,
                ]
                x = tF.raw_input(Shape([2, 3], 2), x_data, dev)
                y = tF.elu(x, k)
                self.assertEqual(Shape([2, 3], 2), y.shape())
                self.assertTrue(np.isclose(y_data, y.to_list()).all())

    def test_TensorForwardTest_CheckSum(self):
        x_data = [
            1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8,
        ]
        shape = [
            Shape([1, 2, 2], 2),
            Shape([2, 1, 2], 2),
            Shape([2, 2], 2),
            Shape([2, 2, 2], 2),
        ]
        y_data = [
            [3, 7, 11, 15, -3, -7, -11, -15],
            [4, 6, 12, 14, -4, -6, -12, -14],
            [6, 8, 10, 12, -6, -8, -10, -12],
            [1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8],
        ]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 2, 2], 2), x_data, dev)
            for i in range(4):
                y = tF.sum(x, i)
                self.assertEqual(shape[i], y.shape())
                self.assertEqual(y_data[i], y.to_list())

    def test_TensorForwardTest_CheckSum2(self):
        ns = [
            1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024, 1025, 65535, 65536, 65537,
        ]
        for dev in TensorForwardTest.devices:
            for n in ns:
                x = tF.constant([n], 1)
                y = tF.sum(x, 0)
                self.assertEqual(Shape(), y.shape())
                self.assertEqual(n, y.to_float())

    def test_TensorForwardTest_CheckLogSumExp(self):
        x_data = [
            1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8,
        ]
        shape = [
            Shape([1, 2, 2], 2),
            Shape([2, 1, 2], 2),
            Shape([2, 2], 2),
            Shape([2, 2, 2], 2),
        ]
        y_data = [
            [2.31326169, 4.31326169, 6.31326169, 8.31326169,
             -0.68673831, -2.68673831, -4.68673831, -6.68673831],
            [3.12692801, 4.12692801, 7.12692801, 8.12692801,
             -0.87307199, -1.87307199, -4.87307199, -5.87307199],
            [5.01814993, 6.01814993, 7.01814993, 8.01814993,
             -0.98185007, -1.98185007, -2.98185007, -3.98185007],
            [1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8],
        ]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 2, 2], 2), x_data, dev)
            for i in range(4):
                y = tF.logsumexp(x, i)
                self.assertEqual(shape[i], y.shape())
                self.assertTrue(np.isclose(y_data[i], y.to_list()).all())

    def test_TensorForwardTest_CheckLogSumExp2(self):
      ns = [
          1, 2, 3, 15, 16, 17, 255, 256, 257, 1023,
          1024, 1025, 65535, 65536, 65537,
      ]
      for dev in TensorForwardTest.devices:
          for n in ns:
              for k in [-5, -1, 0, 1, 5]:
                  x = tF.constant([n], k, dev)
                  y = tF.logsumexp(x, 0)
                  self.assertEqual(Shape(), y.shape())
                  self.assertTrue(np.isclose(
                        [k + math.log(n)], y.to_list(), 0, 1e-3))

    def test_TensorForwardTest_CheckLogSoftmax(self):
        x_data = [
            1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8,
        ]
        y_data = [
            [-1.31326169, -0.31326169, -1.31326169, -0.31326169,
             -1.31326169, -0.31326169, -1.31326169, -0.31326169,
             -0.31326169, -1.31326169, -0.31326169, -1.31326169,
             -0.31326169, -1.31326169, -0.31326169, -1.31326169],
            [-2.12692801, -2.12692801, -0.12692801, -0.12692801,
             -2.12692801, -2.12692801, -0.12692801, -0.12692801,
             -0.12692801, -0.12692801, -2.12692801, -2.12692801,
             -0.12692801, -0.12692801, -2.12692801, -2.12692801],
            [-4.01814993, -4.01814993, -4.01814993, -4.01814993,
             -0.01814993, -0.01814993, -0.01814993, -0.01814993,
             -0.01814993, -0.01814993, -0.01814993, -0.01814993,
             -4.01814993, -4.01814993, -4.01814993, -4.01814993],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 2, 2], 2), x_data, dev)
            for i in range(4):
                y = tF.log_softmax(x, i)
                self.assertEqual(Shape([2, 2, 2], 2), y.shape())
                self.assertTrue(np.isclose(y_data[i], y.to_list(), 0, 1e-6).all())

    def test_TensorForwardTest_CheckLogSoftmax2(self):
        ns = [
            1, 2, 3, 15, 16, 17, 255, 256, 257, 1023,
            1024, 1025, 65535, 65536, 65537,
        ]
        for dev in TensorForwardTest.devices:
            for n in ns:
                for k in [-5, -1, 0, 1, 5]:
                    x = tF.constant([n], k, dev)
                    y = tF.log_softmax(x, 0)
                    self.assertEqual(Shape([n]), y.shape())
                    self.assertTrue(
                      np.isclose([-math.log(n)] * n, y.to_list(), 0, 1e-3).all())

    def test_TensorForwardTest_CheckSoftmax(self):
        x_data = [
          1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8,
        ]
        y_data = [
            [0.26894142, 0.73105858, 0.26894142, 0.73105858,
             0.26894142, 0.73105858, 0.26894142, 0.73105858,
             0.73105858, 0.26894142, 0.73105858, 0.26894142,
             0.73105858, 0.26894142, 0.73105858, 0.26894142],
            [0.11920292, 0.11920292, 0.88079708, 0.88079708,
             0.11920292, 0.11920292, 0.88079708, 0.88079708,
             0.88079708, 0.88079708, 0.11920292, 0.11920292,
             0.88079708, 0.88079708, 0.11920292, 0.11920292],
            [0.01798621, 0.01798621, 0.01798621, 0.01798621,
             0.98201379, 0.98201379, 0.98201379, 0.98201379,
             0.98201379, 0.98201379, 0.98201379, 0.98201379,
             0.01798621, 0.01798621, 0.01798621, 0.01798621],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 2, 2], 2), x_data, dev)
            for i in range(4):
                y = tF.softmax(x, i)
                self.assertEqual(Shape([2, 2, 2], 2), y.shape())
                self.assertTrue(np.isclose(y_data[i], y.to_list(), 0, 1e-6).all())

    def test_TensorForwardTest_CheckSoftmax2(self):
        ns = [
            1, 2, 3, 15, 16, 17, 255, 256, 257, 1023,
            1024, 1025, 65535, 65536, 65537,
        ]
        for dev in TensorForwardTest.devices:
            for n in ns:
                for k in [-5, -1, 0, 1, 5]:
                    x = tF.constant([n], k, dev)
                    y = tF.softmax(x, 0)
                    self.assertEqual(Shape([n]), y.shape())
                    self.assertTrue(
                      np.isclose([1./n] * n, y.to_list(), 0, 1e-6).all())

    def test_TensorForwardTest_CheckBroadcast(self):
        test_cases = [
            (0, 1, Shape([]), [1]),
            (0, 20, Shape([20]), [1] * 20),
            (1, 50, Shape([1, 50]), [1] * 50),
            (2, 100, Shape([1, 1, 100]), [1] * 100),
        ]
        for dev in TensorForwardTest.devices:
            for tc in test_cases:
                x = tF.constant([], 1, dev)
                y = tF.broadcast(x, tc[0], tc[1])
                self.assertEqual(tc[2], y.shape())
                self.assertEqual(tc[3], y.to_list())

    def test_TensorForwardTest_CheckBroadcast2(self):
        test_cases = [
            (1, 1, Shape([2], 3), [1, 2, 3, 4, 5, 6]),
            (2, 1, Shape([2], 3), [1, 2, 3, 4, 5, 6]),
            (1, 2, Shape([2, 2], 3), [1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6]),
            (2, 2, Shape([2, 1, 2], 3), [1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6]),
        ]
        for dev in TensorForwardTest.devices:
            for tc in test_cases:
                x = tF.raw_input(Shape([2], 3), [1, 2, 3, 4, 5, 6], dev)
                y = tF.broadcast(x, tc[0], tc[1])
                self.assertEqual(tc[2], y.shape())
                self.assertEqual(tc[3], y.to_list())

    def test_TensorForwardTest_CheckBroadcast3(self):
        test_cases = [
            (0, 1, Shape([1, 2, 1, 2], 2),
              [1, 2, 3, 4, 5, 6, 7, 8]),
            (2, 1, Shape([1, 2, 1, 2], 2),
              [1, 2, 3, 4, 5, 6, 7, 8]),
            (4, 1, Shape([1, 2, 1, 2], 2),
              [1, 2, 3, 4, 5, 6, 7, 8]),
            (0, 2, Shape([2, 2, 1, 2], 2),
              [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8]),
            (2, 2, Shape([1, 2, 2 ,2], 2),
              [1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6, 7, 8, 7, 8]),
            (4, 2, Shape([1, 2, 1, 2, 2], 2),
              [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8]),
        ]
        for dev in TensorForwardTest.devices:
            for tc in test_cases:
                x = tF.raw_input(
                    Shape([1, 2, 1, 2], 2), [1, 2, 3, 4, 5, 6, 7, 8], dev)
                y = tF.broadcast(x, tc[0], tc[1])
                self.assertEqual(tc[2], y.shape())
                self.assertEqual(tc[3], y.to_list())

    def test_TensorForwardTest_CheckInvalidBroadcast(self):
        for dev in TensorForwardTest.devices:
            x = tF.zeros([1, 2], dev)
            with self.assertRaises(RuntimeError):
                tF.broadcast(x, 0, 0)
            with self.assertRaises(RuntimeError):
                tF.broadcast(x, 1, 0)
            with self.assertRaises(RuntimeError):
                tF.broadcast(x, 1, 1)
            with self.assertRaises(RuntimeError):
                tF.broadcast(x, 1, 3)
            with self.assertRaises(RuntimeError):
                tF.broadcast(x, 2, 0)

    def test_TensorForwardTest_CheckBatchSum(self):
        x_data = [
            1, 2, 3, 4, 5, 6, 7, 8,
            -2, -4, -6, -8, -10, -12, -14, -16,
        ]
        y_data = [
            -1, -2, -3, -4, -5, -6, -7, -8,
        ]
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 2, 2], 2), x_data, dev)
            y = tF.batch.sum(x)
            self.assertEqual(Shape([2, 2, 2]), y.shape())
            self.assertEqual(y_data, y.to_list())

    def test_TensorForwardTest_CheckSoftmaxCrossEntropy(self):
        x_data = [
            [-1, 0, 1, 1, 0, 0, 0, 0, 1],
            [-1, 1, 0, 0, 0, 0, 1, 0, 1],
        ]
        t_data = [
            [1./3, 1./3, 1./3, .5, .25, .25, 0, 0, 1],
            [1./3, .5, 0, 1./3, .25, 0, 1./3, .25, 1],
        ]
        y_data = [
            [1.40760596, 1.05144471, 0.55144471],
            [1.40760596, 1.05144471, 0.55144471],
        ]
        shape = [Shape([1, 3]), Shape([3])]
        for dev in TensorForwardTest.devices:
            for dim in [0, 1]:
                x = tF.raw_input([3, 3], x_data[dim], dev)
                t = tF.raw_input([3, 3], t_data[dim], dev)
                y = tF.softmax_cross_entropy(x, t, dim)
                self.assertEqual(shape[dim], y.shape())
                self.assertTrue(np.isclose(y_data[dim], y.to_list()).all())

    def test_TensorForwardTest_CheckSoftmaxCrossEntropyBatchBroadcast(self):
        test_cases = [
            ([-1, 0, 1],
             [1, 0, 0, 0, 1, 0, 0, 0, 1],
             [2.40760596, 1.40760596, 0.40760596],
             Shape([3]), Shape([3], 3), Shape([], 3)),
            ([-1, 0, 1, 1, -1, 0, 0, 1, -1],
             [1, 0, 0],
             [2.40760596, 0.40760596, 1.40760596],
             Shape([3], 3), Shape([3]), Shape([], 3)),
        ]
        for dev in TensorForwardTest.devices:
            for tc in test_cases:
                x = tF.raw_input(tc[3], tc[0], dev)
                t = tF.raw_input(tc[4], tc[1], dev)
                y = tF.softmax_cross_entropy(x, t, 0)
                self.assertEqual(tc[5], y.shape())
                self.assertTrue(np.isclose(tc[2], y.to_list()).all())

    def test_TensorForwardTest_CheckInvalidSoftmaxCrossEntropy(self):
        for dev in TensorForwardTest.devices:
            x = tF.constant([2, 2], .5, dev)
            t = tF.constant([2, 3], .5, dev)
            with self.assertRaises(RuntimeError):
                tF.softmax_cross_entropy(x, t, 0)
            with self.assertRaises(RuntimeError):
                tF.softmax_cross_entropy(x, t, 1)
            with self.assertRaises(RuntimeError):
                tF.softmax_cross_entropy(x, t, 2)
            x = tF.constant(Shape([2, 2], 2), .5, dev)
            t = tF.constant(Shape([2, 3], 3), .5, dev)
            with self.assertRaises(RuntimeError):
                tF.softmax_cross_entropy(x, t, 0)
            with self.assertRaises(RuntimeError):
                tF.softmax_cross_entropy(x, t, 1)
            with self.assertRaises(RuntimeError):
                tF.softmax_cross_entropy(x, t, 2)

    def test_TensorForwardTest_CheckSparseSoftmaxCrossEntropy(self):
        test_cases = [
            ([-1, 0, 1, 1, -1, 0, 0, 1, -1],
             0, [0], Shape([3, 3]), Shape([1, 3]),
             [2.40760596, 0.40760596, 1.40760596]),
            ([-1, 0, 1, 1, -1, 0, 0, 1, -1],
             1, [1], Shape([3, 3]), Shape([3]),
             [0.40760596, 2.40760596, 1.40760596]),
            ([-1, 0, 1, 1, -1, 0, 0, 1, -1],
             2, [0], Shape([3, 3]), Shape([3, 3]),
             [0, 0, 0, 0, 0, 0, 0, 0, 0]),
            ([-1, 0, 1, 1, -1, 0, 0, 1, -1, -2, 0, 2, 2, -2, 0, 0, 2, -2],
             0, [0, 1], Shape([3, 3], 2), Shape([1, 3], 2),
             [2.40760596, 0.40760596, 1.40760596,
              2.14293163, 4.14293163, 0.14293163]),
            ([-1, 0, 1, 1, -1, 0, 0, 1, -1, -2, 0, 2, 2, -2, 0, 0, 2, -2],
             0, [0], Shape([3, 3], 2), Shape([1, 3], 2),
             [2.40760596, 0.40760596, 1.40760596,
              4.14293163, 0.14293163, 2.14293163]),
            ([-1, 0, 1, 1, -1, 0, 0, 1, -1],
             0, [0, 1], Shape([3, 3]), Shape([1, 3], 2),
             [2.40760596, 0.40760596, 1.40760596,
              1.40760596, 2.40760596, 0.40760596]),
        ]
        for dev in TensorForwardTest.devices:
            for tc in test_cases:
                x = tF.raw_input(tc[3], tc[0], dev)
                y = tF.softmax_cross_entropy(x, tc[2], tc[1])
                self.assertEqual(tc[4], y.shape())
                self.assertTrue(np.isclose(tc[5], y.to_list(), 0, 1e-6).all())

    def test_TensorForwardTest_CheckInvalidSparseSoftmaxCrossEntropy(self):
        for dev in TensorForwardTest.devices:
            x = tF.constant([2, 2], .5, dev)
            t = [2]
            with self.assertRaises(RuntimeError):
                tF.softmax_cross_entropy(x, t, 0)
            with self.assertRaises(RuntimeError):
                tF.softmax_cross_entropy(x, t, 1)
            with self.assertRaises(RuntimeError):
                tF.softmax_cross_entropy(x, t, 2)
            x = tF.constant(Shape([2, 2], 2), .5, dev)
            t = [0, 0, 0]
            with self.assertRaises(RuntimeError):
                tF.softmax_cross_entropy(x, t, 0)
            with self.assertRaises(RuntimeError):
                tF.softmax_cross_entropy(x, t, 1)
            with self.assertRaises(RuntimeError):
                tF.softmax_cross_entropy(x, t, 2)

    def test_TensorForwardTest_CheckStopGradient(self):
        x_data = [
            0, .5, 1, 2, 3, 4,
            0, -.5, -1, -2, -3, -4,
        ]
        y_data = x_data
        for dev in TensorForwardTest.devices:
            x = tF.raw_input(Shape([2, 3], 2), x_data, dev)
            y = tF.stop_gradient(x)
            self.assertEqual(Shape([2, 3], 2), y.shape())
            self.assertTrue(np.isclose(y_data, y.to_list()).all())

    def run_test_conv2d(self, x_shape, x_data, w_shape, w_data, y_shape, y_data, pad0, pad1, str0, str1, dil0, dil1):
        for dev in TensorForwardTest.devices:
            try:
                x = tF.raw_input(x_shape, x_data, dev)
                w = tF.raw_input(w_shape, w_data, dev)
                y = tF.conv2d(x, w, pad0, pad1, str0, str1, dil0, dil1)
                self.assertEqual(y_shape, y.shape())
                self.assertTrue(np.isclose(y_data, y.to_list()).all)
            except RuntimeError as e:
                # TODO(vbkaisetsu):
                # We have to implement a better method to detect
                # NotImplementedError in Python
                if "Not implemented" not in str(e):
                    raise

    def test_TensorForwardTest_CheckConv2D_1x1x1_1x1x1x1(self):
        x_data = [123]
        w_data = [42]
        y_data = [123 * 42]
        x_shape = Shape([])
        w_shape = Shape([])
        y_shape = Shape([])
        self.run_test_conv2d(x_shape, x_data, w_shape, w_data, y_shape, y_data, 0, 0, 1, 1, 1, 1)

    def test_TensorForwardTest_CheckConv2D_5x1x1_1x1x1x1(self):
        x_data = list(range(1, 5 + 1))
        w_data = [42]
        y_data = [42, 84, 126, 168, 210]
        x_shape = Shape([5])
        w_shape = Shape([])
        y_shape = Shape([5])
        self.run_test_conv2d(x_shape, x_data, w_shape, w_data, y_shape, y_data, 0, 0, 1, 1, 1, 1)

    def test_TensorForwardTest_CheckConv2D_5x1x1_2x1x1x1(self):
        x_data = list(range(1, 5 + 1))
        w_data = list(range(1, 2 + 1))
        y_data = [4, 7, 10, 13]
        x_shape = Shape([5])
        w_shape = Shape([2])
        y_shape = Shape([4])
        self.run_test_conv2d(x_shape, x_data, w_shape, w_data, y_shape, y_data, 0, 0, 1, 1, 1, 1)

    def test_TensorForwardTest_CheckConv2D_5x1x1_5x1x1x1(self):
        x_data = list(range(1, 5 + 1))
        w_data = list(range(1, 5 + 1))
        y_data = [35]
        x_shape = Shape([5])
        w_shape = Shape([5])
        y_shape = Shape([])
        self.run_test_conv2d(x_shape, x_data, w_shape, w_data, y_shape, y_data, 0, 0, 1, 1, 1, 1)

    def test_TensorForwardTest_CheckConv2D_1x5x1_1x1x1x1(self):
        x_data = list(range(1, 5 + 1))
        w_data = [42]
        y_data = [42, 84, 126, 168, 210]
        x_shape = Shape([1, 5])
        w_shape = Shape([])
        y_shape = Shape([1, 5])
        self.run_test_conv2d(x_shape, x_data, w_shape, w_data, y_shape, y_data, 0, 0, 1, 1, 1, 1)

    def test_TensorForwardTest_CheckConv2D_1x5x1_1x2x1x1(self):
        x_data = list(range(1, 5 + 1))
        w_data = list(range(1, 2 + 1))
        y_data = [4, 7, 10, 13]
        x_shape = Shape([1, 5])
        w_shape = Shape([1, 2])
        y_shape = Shape([1, 4])
        self.run_test_conv2d(x_shape, x_data, w_shape, w_data, y_shape, y_data, 0, 0, 1, 1, 1, 1)

    def test_TensorForwardTest_CheckConv2D_1x5x1_1x5x1x1(self):
        x_data = list(range(1, 5 + 1))
        w_data = list(range(1, 5 + 1))
        y_data = [35]
        x_shape = Shape([1, 5])
        w_shape = Shape([1, 5])
        y_shape = Shape([])
        self.run_test_conv2d(x_shape, x_data, w_shape, w_data, y_shape, y_data, 0, 0, 1, 1, 1, 1)

    def test_TensorForwardTest_CheckConv2D_5x5x1_1x1x1x1(self):
        x_data = list(range(1, 5 * 5 + 1))
        w_data = [42]
        y_data = [
             42,  84, 126,  168,  210,
            252, 294, 336,  378,  420,
            462, 504, 546,  588,  630,
            672, 714, 756,  798,  840,
            882, 924, 966, 1008, 1050,
        ]
        x_shape = Shape([5, 5])
        w_shape = Shape([])
        y_shape = Shape([5, 5])
        self.run_test_conv2d(x_shape, x_data, w_shape, w_data, y_shape, y_data, 0, 0, 1, 1, 1, 1)

    def test_TensorForwardTest_CheckConv2D_5x5x1_2x1x1x1(self):
        x_data = list(range(1, 5 * 5 + 1))
        w_data = list(range(1, 2 + 1))
        y_data = [
             4,  7, 10, 13,
            19, 22, 25, 28,
            34, 37, 40, 43,
            49, 52, 55, 58,
            64, 67, 70, 73,
        ]
        x_shape = Shape([5, 5])
        w_shape = Shape([2])
        y_shape = Shape([4, 5])
        self.run_test_conv2d(x_shape, x_data, w_shape, w_data, y_shape, y_data, 0, 0, 1, 1, 1, 1)

    def test_TensorForwardTest_CheckConv2D_5x5x1_5x1x1x1(self):
        x_data = list(range(1, 5 * 5 + 1))
        w_data = list(range(1, 5 + 1))
        y_data = [
             35,
            110,
            185,
            260,
            335,
        ]
        x_shape = Shape([5, 5])
        w_shape = Shape([5])
        y_shape = Shape([1, 5])
        self.run_test_conv2d(x_shape, x_data, w_shape, w_data, y_shape, y_data, 0, 0, 1, 1, 1, 1)

    def test_TensorForwardTest_CheckConv2D_5x5x1_1x2x1x1(self):
        x_data = list(range(1, 5 * 5 + 1))
        w_data = list(range(1, 2 + 1))
        y_data = [
             8, 11, 14, 17, 20,
            23, 26, 29, 32, 35,
            38, 41, 44, 47, 50,
            53, 56, 59, 62, 65,
        ]
        x_shape = Shape([5, 5])
        w_shape = Shape([1, 2])
        y_shape = Shape([5, 4])
        self.run_test_conv2d(x_shape, x_data, w_shape, w_data, y_shape, y_data, 0, 0, 1, 1, 1, 1)

    def test_TensorForwardTest_CheckConv2D_5x5x1_2x2x1x1(self):
        x_data = list(range(1, 5 * 5 + 1))
        w_data = list(range(1, 2 * 2 + 1))
        y_data = [
             29,  39,  49,  59,
             79,  89,  99, 109,
            129, 139, 149, 159,
            179, 189, 199, 209,
        ]
        x_shape = Shape([5, 5])
        w_shape = Shape([2, 2])
        y_shape = Shape([4, 4])
        self.run_test_conv2d(x_shape, x_data, w_shape, w_data, y_shape, y_data, 0, 0, 1, 1, 1, 1)

    def test_TensorForwardTest_CheckConv2D_5x5x1_5x2x1x1(self):
        x_data = list(range(1, 5 * 5 + 1))
        w_data = list(range(1, 5 * 2 + 1))
        y_data = [
             220,
             495,
             770,
            1045,
        ]
        x_shape = Shape([5, 5])
        w_shape = Shape([5, 2])
        y_shape = Shape([1, 4])
        self.run_test_conv2d(x_shape, x_data, w_shape, w_data, y_shape, y_data, 0, 0, 1, 1, 1, 1)

    def test_TensorForwardTest_CheckConv2D_5x5x1_1x5x1x1(self):
        x_data = list(range(1, 5 * 5 + 1))
        w_data = list(range(1, 1 * 5 + 1))
        y_data = [
            115, 130, 145, 160, 175,
        ]
        x_shape = Shape([5, 5])
        w_shape = Shape([1, 5])
        y_shape = Shape([5])
        self.run_test_conv2d(x_shape, x_data, w_shape, w_data, y_shape, y_data, 0, 0, 1, 1, 1, 1)

    def test_TensorForwardTest_CheckConv2D_5x5x1_2x5x1x1(self):
        x_data = list(range(1, 5 * 5 + 1))
        w_data = list(range(1, 2 * 5 + 1))
        y_data = [
            430, 485, 540, 595,
        ]
        x_shape = Shape([5, 5])
        w_shape = Shape([2, 5])
        y_shape = Shape([4])
        self.run_test_conv2d(x_shape, x_data, w_shape, w_data, y_shape, y_data, 0, 0, 1, 1, 1, 1)

    def test_TensorForwardTest_CheckConv2D_5x5x1_5x5x1x1(self):
        x_data = list(range(1, 5 * 5 + 1))
        w_data = list(range(1, 5 * 5 + 1))
        y_data = [2925]
        x_shape = Shape([5, 5])
        w_shape = Shape([5, 5])
        y_shape = Shape([])
        self.run_test_conv2d(x_shape, x_data, w_shape, w_data, y_shape, y_data, 0, 0, 1, 1, 1, 1)

    def test_TensorForwardTest_CheckConv2D_5x5x3_2x2x3x1(self):
        x_data = list(range(1, 5 * 5 * 3 + 1))
        w_data = list(range(1, 2 * 2 * 3 + 1))
        y_data = [
            3029, 3107, 3185, 3263,
            3419, 3497, 3575, 3653,
            3809, 3887, 3965, 4043,
            4199, 4277, 4355, 4433,
        ]
        x_shape = Shape([5, 5, 3])
        w_shape = Shape([2, 2, 3])
        y_shape = Shape([4, 4])
        self.run_test_conv2d(x_shape, x_data, w_shape, w_data, y_shape, y_data, 0, 0, 1, 1, 1, 1)

    def test_TensorForwardTest_CheckConv2D_5x5x1_2x2x1x3(self):
        x_data = list(range(1, 5 * 5 + 1))
        w_data = list(range(1, 2 * 2 * 3 + 1))
        y_data = [
            # channel 1
             29,  39,  49,  59,
             79,  89,  99, 109,
            129, 139, 149, 159,
            179, 189, 199, 209,
            # channel 2
             93, 119, 145, 171,
            223, 249, 275, 301,
            353, 379, 405, 431,
            483, 509, 535, 561,
            # channel 3
            157, 199, 241, 283,
            367, 409, 451, 493,
            577, 619, 661, 703,
            787, 829, 871, 913,
        ]
        x_shape = Shape([5, 5])
        w_shape = Shape([2, 2, 1, 3])
        y_shape = Shape([4, 4, 3])
        self.run_test_conv2d(x_shape, x_data, w_shape, w_data, y_shape, y_data, 0, 0, 1, 1, 1, 1)

    def test_TensorForwardTest_CheckConv2D_5x5x3_2x2x3x3(self):
        x_data = list(range(1, 5 * 5 * 3 + 1))
        w_data = list(range(1, 2 * 2 * 3 * 3 + 1))
        y_data = [
            # channel 1
            3029, 3107, 3185, 3263,
            3419, 3497, 3575, 3653,
            3809, 3887, 3965, 4043,
            4199, 4277, 4355, 4433,
            # channel 2
             7205,  7427,  7649,  7871,
             8315,  8537,  8759,  8981,
             9425,  9647,  9869, 10091,
            10535, 10757, 10979, 11201,
            # channel 3
            11381, 11747, 12113, 12479,
            13211, 13577, 13943, 14309,
            15041, 15407, 15773, 16139,
            16871, 17237, 17603, 17969,
        ]
        x_shape = Shape([5, 5, 3])
        w_shape = Shape([2, 2, 3, 3])
        y_shape = Shape([4, 4, 3])
        self.run_test_conv2d(x_shape, x_data, w_shape, w_data, y_shape, y_data, 0, 0, 1, 1, 1, 1)

    def test_TensorForwardTest_CheckConv2D_5x5x1_2x2x1x1_Padding10(self):
        x_data = list(range(1, 5 * 5 + 1))
        w_data = list(range(1, 2 * 2 + 1))
        y_data = [
             9,  29,  39,  49,  59,  40,
            29,  79,  89,  99, 109,  70,
            49, 129, 139, 149, 159, 100,
            69, 179, 189, 199, 209, 130,
        ]
        x_shape = Shape([5, 5])
        w_shape = Shape([2, 2])
        y_shape = Shape([6, 4])
        self.run_test_conv2d(x_shape, x_data, w_shape, w_data, y_shape, y_data, 1, 0, 1, 1, 1, 1)

    def test_TensorForwardTest_CheckConv2D_5x5x1_2x2x1x1_Padding01(self):
        x_data = list(range(1, 5 * 5 + 1))
        w_data = list(range(1, 2 * 2 + 1))
        y_data = [
              4,   7,  10,  13,
             29,  39,  49,  59,
             79,  89,  99, 109,
            129, 139, 149, 159,
            179, 189, 199, 209,
            150, 157, 164, 171,
        ]
        x_shape = Shape([5, 5])
        w_shape = Shape([2, 2])
        y_shape = Shape([4, 6])
        self.run_test_conv2d(x_shape, x_data, w_shape, w_data, y_shape, y_data, 0, 1, 1, 1, 1, 1)

    def test_TensorForwardTest_CheckConv2D_5x5x1_2x2x1x1_Padding11(self):
        x_data = list(range(1, 5 * 5 + 1))
        w_data = list(range(1, 2 * 2 + 1))
        y_data = [
             1,   4,   7,  10,  13,  10,
             9,  29,  39,  49,  59,  40,
            29,  79,  89,  99, 109,  70,
            49, 129, 139, 149, 159, 100,
            69, 179, 189, 199, 209, 130,
            63, 150, 157, 164, 171, 100,
        ]
        x_shape = Shape([5, 5])
        w_shape = Shape([2, 2])
        y_shape = Shape([6, 6])
        self.run_test_conv2d(x_shape, x_data, w_shape, w_data, y_shape, y_data, 1, 1, 1, 1, 1, 1)

    def test_TensorForwardTest_CheckConv2D_5x5x1_2x2x1x1_Stride21(self):
        x_data = list(range(1, 5 * 5 + 1))
        w_data = list(range(1, 2 * 2 + 1))
        y_data = [
             29,  49,
             79,  99,
            129, 149,
            179, 199,
        ]
        x_shape = Shape([5, 5])
        w_shape = Shape([2, 2])
        y_shape = Shape([2, 4])
        self.run_test_conv2d(x_shape, x_data, w_shape, w_data, y_shape, y_data, 0, 0, 2, 1, 1, 1)

    def test_TensorForwardTest_CheckConv2D_5x5x1_2x2x1x1_Stride12(self):
        x_data = list(range(1, 5 * 5 + 1))
        w_data = list(range(1, 2 * 2 + 1))
        y_data = [
             29,  39,  49,  59,
            129, 139, 149, 159,
        ]
        x_shape = Shape([5, 5])
        w_shape = Shape([2, 2])
        y_shape = Shape([4, 2])
        self.run_test_conv2d(x_shape, x_data, w_shape, w_data, y_shape, y_data, 0, 0, 1, 2, 1, 1)

    def test_TensorForwardTest_CheckConv2D_5x5x1_2x2x1x1_Stride22(self):
        x_data = list(range(1, 5 * 5 + 1))
        w_data = list(range(1, 2 * 2 + 1))
        y_data = [
             29,  49,
            129, 149,
        ]
        x_shape = Shape([5, 5])
        w_shape = Shape([2, 2])
        y_shape = Shape([2, 2])
        self.run_test_conv2d(x_shape, x_data, w_shape, w_data, y_shape, y_data, 0, 0, 2, 2, 1, 1)

    def test_TensorForwardTest_CheckConv2D_5x5x1_2x2x1x1_Dilation21(self):
        x_data = list(range(1, 5 * 5 + 1))
        w_data = list(range(1, 2 * 2 + 1))
        y_data = [
             33,  43,  53,
             83,  93, 103,
            133, 143, 153,
            183, 193, 203,
        ]
        x_shape = Shape([5, 5])
        w_shape = Shape([2, 2])
        y_shape = Shape([3, 4])
        self.run_test_conv2d(x_shape, x_data, w_shape, w_data, y_shape, y_data, 0, 0, 1, 1, 2, 1)

    def test_TensorForwardTest_CheckConv2D_5x5x1_2x2x1x1_Dilation12(self):
        x_data = list(range(1, 5 * 5 + 1))
        w_data = list(range(1, 2 * 2 + 1))
        y_data = [
             44,  54,  64,  74,
             94, 104, 114, 124,
            144, 154, 164, 174,
        ]
        x_shape = Shape([5, 5])
        w_shape = Shape([2, 2])
        y_shape = Shape([4, 3])
        self.run_test_conv2d(x_shape, x_data, w_shape, w_data, y_shape, y_data, 0, 0, 1, 1, 1, 2)

    def test_TensorForwardTest_CheckConv2D_5x5x1_2x2x1x1_Dilation22(self):
        x_data = list(range(1, 5 * 5 + 1))
        w_data = list(range(1, 2 * 2 + 1))
        y_data = [
             48,  58,  68,
             98, 108, 118,
            148, 158, 168,
        ]
        x_shape = Shape([5, 5])
        w_shape = Shape([2, 2])
        y_shape = Shape([3, 3])
        self.run_test_conv2d(x_shape, x_data, w_shape, w_data, y_shape, y_data, 0, 0, 1, 1, 2, 2)

    def test_TensorForwardTest_CheckConv2D_5x5x1_2x2x1x1_N1(self):
        x_data = list(range(1, 5 * 5 * 3 + 1))
        w_data = list(range(1, 2 * 2 + 1))
        y_data = [
            # minibatch 1
             29,  39,  49,  59,
             79,  89,  99, 109,
            129, 139, 149, 159,
            179, 189, 199, 209,
            # minibatch 2
            279, 289, 299, 309,
            329, 339, 349, 359,
            379, 389, 399, 409,
            429, 439, 449, 459,
            # minibatch 3
            529, 539, 549, 559,
            579, 589, 599, 609,
            629, 639, 649, 659,
            679, 689, 699, 709,
        ]
        x_shape = Shape([5, 5], 3)
        w_shape = Shape([2, 2])
        y_shape = Shape([4, 4], 3)
        self.run_test_conv2d(x_shape, x_data, w_shape, w_data, y_shape, y_data, 0, 0, 1, 1, 1, 1)

    def test_TensorForwardTest_CheckConv2D_5x5x1_2x2x1x1_1N(self):
        x_data = list(range(1, 5 * 5 + 1))
        w_data = list(range(1, 2 * 2 * 3 + 1))
        y_data = [
            # minibatch 1
             29,  39,  49,  59,
             79,  89,  99, 109,
            129, 139, 149, 159,
            179, 189, 199, 209,
            # minibatch 2
             93, 119, 145, 171,
            223, 249, 275, 301,
            353, 379, 405, 431,
            483, 509, 535, 561,
            # minibatch 3
            157, 199, 241, 283,
            367, 409, 451, 493,
            577, 619, 661, 703,
            787, 829, 871, 913,
        ]
        x_shape = Shape([5, 5])
        w_shape = Shape([2, 2], 3)
        y_shape = Shape([4, 4], 3)
        self.run_test_conv2d(x_shape, x_data, w_shape, w_data, y_shape, y_data, 0, 0, 1, 1, 1, 1)

    def test_TensorForwardTest_CheckConv2D_5x5x1_2x2x1x1_NN(self):
        x_data = list(range(1, 5 * 5 * 3 + 1))
        w_data = list(range(1, 2 * 2 * 3 + 1))
        y_data = [
            # minibatch 1
             29,  39,  49,  59,
             79,  89,  99, 109,
            129, 139, 149, 159,
            179, 189, 199, 209,
            # minibatch 2
             743,  769,  795,  821,
             873,  899,  925,  951,
            1003, 1029, 1055, 1081,
            1133, 1159, 1185, 1211,
            # minibatch 3
            2257, 2299, 2341, 2383,
            2467, 2509, 2551, 2593,
            2677, 2719, 2761, 2803,
            2887, 2929, 2971, 3013,
        ]
        x_shape = Shape([5, 5], 3)
        w_shape = Shape([2, 2], 3)
        y_shape = Shape([4, 4], 3)
        self.run_test_conv2d(x_shape, x_data, w_shape, w_data, y_shape, y_data, 0, 0, 1, 1, 1, 1)

    def test_TensorForwardTest_CheckConv2D_VGG16FirstLayer(self):
        x_data = [1] * (224 * 224 * 3)
        w_data = [1] * (3 * 3 * 3 * 64)
        y_data = [27] * (224 * 224 * 64)
        for b in range(64):
            y_data[0 + b * 224 * 224] += 3;
            y_data[223 + b * 224 * 224] += 3;
            y_data[223 * 224 + b * 224 * 224] += 3;
            y_data[223 * 224 + 223 + b * 224 * 224] += 3;
            for i in range(224):
                y_data[i + b * 224 * 224] -= 3 * 3
                y_data[223 * 224 + i + b * 224 * 224] -= 3 * 3
                y_data[i * 224 + b * 224 * 224] -= 3 * 3
                y_data[i * 224 + 223 + b * 224 * 224] -= 3 * 3

        x_shape = Shape([224, 224, 3])
        w_shape = Shape([3, 3, 3, 64])
        y_shape = Shape([224, 224, 64])
        self.run_test_conv2d(x_shape, x_data, w_shape, w_data, y_shape, y_data, 1, 1, 1, 1, 1, 1)

    def test_TensorForwardTest_CheckInvalidConv2D(self):
        test_cases = [
            # invalid #dimensions
            (Shape([1, 1, 1, 2]), Shape([]), 0, 0, 1, 1, 1, 1, False),
            (Shape([]), Shape([1, 1, 1, 1, 2]), 0, 0, 1, 1, 1, 1, False),
            # zero-stride/dilation
            (Shape([]), Shape([]), 0, 0, 1, 1, 1, 1, True),
            (Shape([]), Shape([]), 0, 0, 0, 1, 1, 1, False),
            (Shape([]), Shape([]), 0, 0, 1, 0, 1, 1, False),
            (Shape([]), Shape([]), 0, 0, 1, 1, 0, 1, False),
            (Shape([]), Shape([]), 0, 0, 1, 1, 1, 0, False),
            # minibatches mismatching
            (Shape([], 2), Shape([], 2), 0, 0, 1, 1, 1, 1, True),
            (Shape([], 3), Shape([], 3), 0, 0, 1, 1, 1, 1, True),
            (Shape([], 2), Shape([], 3), 0, 0, 1, 1, 1, 1, False),
            # channels mismatching
            (Shape([3, 3, 42]), Shape([3, 3, 42]), 0, 0, 1, 1, 1, 1, True),
            (Shape([3, 3, 42]), Shape([3, 3, 43]), 0, 0, 1, 1, 1, 1, False),
            # sizes mismatching
            (Shape([3, 3]), Shape([3, 3]), 0, 0, 1, 1, 1, 1, True),
            (Shape([3, 3]), Shape([4, 3]), 0, 0, 1, 1, 1, 1, False),
            (Shape([3, 3]), Shape([3, 4]), 0, 0, 1, 1, 1, 1, False),
            (Shape([3, 3]), Shape([4, 4]), 0, 0, 1, 1, 1, 1, False),
            # sizes mismatching with padding
            (Shape([3, 3]), Shape([5, 5]), 1, 1, 1, 1, 1, 1, True),
            (Shape([3, 3]), Shape([6, 5]), 1, 1, 1, 1, 1, 1, False),
            (Shape([3, 3]), Shape([5, 6]), 1, 1, 1, 1, 1, 1, False),
            (Shape([3, 3]), Shape([6, 6]), 1, 1, 1, 1, 1, 1, False),
            # sizes mismatching with stride
            (Shape([3, 3]), Shape([3, 3]), 0, 0, 2, 2, 1, 1, True),
            (Shape([3, 3]), Shape([4, 3]), 0, 0, 2, 2, 1, 1, False),
            (Shape([3, 3]), Shape([3, 4]), 0, 0, 2, 2, 1, 1, False),
            (Shape([3, 3]), Shape([4, 4]), 0, 0, 2, 2, 1, 1, False),
            # sizes mismatching with dilation
            (Shape([3, 3]), Shape([2, 2]), 0, 0, 1, 1, 2, 2, True),
            (Shape([2, 3]), Shape([2, 2]), 0, 0, 1, 1, 2, 2, False),
            (Shape([3, 2]), Shape([2, 2]), 0, 0, 1, 1, 2, 2, False),
            (Shape([2, 2]), Shape([2, 2]), 0, 0, 1, 1, 2, 2, False),
            (Shape([3, 3]), Shape([2, 2]), 0, 0, 1, 1, 3, 2, False),
            (Shape([3, 3]), Shape([2, 2]), 0, 0, 1, 1, 2, 3, False),
            (Shape([3, 3]), Shape([2, 2]), 0, 0, 1, 1, 3, 3, False),
        ]

        for dev in TensorForwardTest.devices:
            for tc in test_cases:
                x = tF.constant(tc[0], 0, dev)
                w = tF.constant(tc[1], 0, dev)
                if tc[8]:
                    try:
                        tF.conv2d(x, w, tc[2], tc[3], tc[4], tc[5], tc[6], tc[7])
                    except RuntimeError as e:
                        # TODO(vbkaisetsu):
                        # We have to implement a better method to detect
                        # NotImplementedError in Python
                        if "Not implemented" not in str(e):
                            raise
                else:
                    with self.assertRaises(RuntimeError) as e:
                        tF.conv2d(x, w, tc[2], tc[3], tc[4], tc[5], tc[6], tc[7])

    def run_test_max_pool2d(self, x_shape, x_data, y_shape, y_data, win0, win1, pad0, pad1, str0, str1):
        for dev in TensorForwardTest.devices:
            try:
                print(dev)
                x = tF.raw_input(x_shape, x_data, dev)
                y = tF.max_pool2d(x, win0, win1, pad0, pad1, str0, str1)
                self.assertEqual(y_shape, y.shape())
                self.assertTrue(np.isclose(y_data, y.to_list()).all)
            except RuntimeError as e:
                # TODO(vbkaisetsu):
                # We have to implement a better method to detect
                # NotImplementedError in Python
                if "Not implemented" not in str(e):
                    raise

    def test_TensorForwardTest_CheckMaxPool2D_1x1x1_1x1(self):
        x_data = [123]
        y_data = [123]
        x_shape = Shape([])
        y_shape = Shape([])
        self.run_test_max_pool2d(x_shape, x_data, y_shape, y_data, 1, 1, 0, 0, 1, 1)

    def test_TensorForwardTest_CheckMaxPool2D_5x1x1_1x1(self):
        x_data = list(range(1, 5 + 1))
        y_data = [1, 2, 3, 4, 5]
        x_shape = Shape([5])
        y_shape = Shape([5])
        self.run_test_max_pool2d(x_shape, x_data, y_shape, y_data, 1, 1, 0, 0, 1, 1)

    def test_TensorForwardTest_CheckMaxPool2D_5x1x1_2x1(self):
        x_data = list(range(1, 5 + 1))
        y_data = [2, 3, 4, 5]
        x_shape = Shape([5])
        y_shape = Shape([4])
        self.run_test_max_pool2d(x_shape, x_data, y_shape, y_data, 2, 1, 0, 0, 1, 1)

    def test_TensorForwardTest_CheckMaxPool2D_5x1x1_5x1(self):
        x_data = list(range(1, 5 + 1))
        y_data = [5]
        x_shape = Shape([5])
        y_shape = Shape([])
        self.run_test_max_pool2d(x_shape, x_data, y_shape, y_data, 5, 1, 0, 0, 1, 1)

    def test_TensorForwardTest_CheckMaxPool2D_1x5x1_1x1(self):
        x_data = list(range(1, 5 + 1))
        y_data = [1, 2, 3, 4, 5]
        x_shape = Shape([1, 5])
        y_shape = Shape([1, 5])
        self.run_test_max_pool2d(x_shape, x_data, y_shape, y_data, 1, 1, 0, 0, 1, 1)

    def test_TensorForwardTest_CheckMaxPool2D_1x5x1_1x2(self):
        x_data = list(range(1, 5 + 1))
        y_data = [2, 3, 4, 5]
        x_shape = Shape([1, 5])
        y_shape = Shape([1, 4])
        self.run_test_max_pool2d(x_shape, x_data, y_shape, y_data, 1, 2, 0, 0, 1, 1)

    def test_TensorForwardTest_CheckMaxPool2D_1x5x1_1x5(self):
        x_data = list(range(1, 5 + 1))
        y_data = [5]
        x_shape = Shape([1, 5])
        y_shape = Shape([])
        self.run_test_max_pool2d(x_shape, x_data, y_shape, y_data, 1, 5, 0, 0, 1, 1)

    def test_TensorForwardTest_CheckMaxPool2D_5x5x1_1x1(self):
        x_data = list(range(1, 5 * 5 + 1))
        y_data = [
             1,  2,  3,  4,  5,
             6,  7,  8,  9, 10,
            11, 12, 13, 14, 15,
            16, 17, 18, 19, 20,
            21, 22, 23, 24, 25,
        ]
        x_shape = Shape([5, 5])
        y_shape = Shape([5, 5])
        self.run_test_max_pool2d(x_shape, x_data, y_shape, y_data, 1, 1, 0, 0, 1, 1)

    def test_TensorForwardTest_CheckMaxPool2D_5x5x1_2x1(self):
        x_data = list(range(1, 5 * 5 + 1))
        y_data = [
             2,  3,  4,  5,
             7,  8,  9, 10,
            12, 13, 14, 15,
            17, 18, 19, 20,
            22, 23, 24, 25,
        ]
        x_shape = Shape([5, 5])
        y_shape = Shape([4, 5])
        self.run_test_max_pool2d(x_shape, x_data, y_shape, y_data, 2, 1, 0, 0, 1, 1)

    def test_TensorForwardTest_CheckMaxPool2D_5x5x1_5x1(self):
        x_data = list(range(1, 5 * 5 + 1))
        y_data = [
             5,
            10,
            15,
            20,
            25,
        ]
        x_shape = Shape([5, 5])
        y_shape = Shape([1, 5])
        self.run_test_max_pool2d(x_shape, x_data, y_shape, y_data, 5, 1, 0, 0, 1, 1)

    def test_TensorForwardTest_CheckMaxPool2D_5x5x1_1x2(self):
        x_data = list(range(1, 5 * 5 + 1))
        y_data = [
             6,  7,  8,  9, 10,
            11, 12, 13, 14, 15,
            16, 17, 18, 19, 20,
            21, 22, 23, 24, 25,
        ]
        x_shape = Shape([5, 5])
        y_shape = Shape([5, 4])
        self.run_test_max_pool2d(x_shape, x_data, y_shape, y_data, 1, 2, 0, 0, 1, 1)

    def test_TensorForwardTest_CheckMaxPool2D_5x5x1_2x2(self):
        x_data = list(range(1, 5 * 5 + 1))
        y_data = [
             7,  8,  9, 10,
            12, 13, 14, 15,
            17, 18, 19, 20,
            22, 23, 24, 25,
        ]
        x_shape = Shape([5, 5])
        y_shape = Shape([4, 4])
        self.run_test_max_pool2d(x_shape, x_data, y_shape, y_data, 2, 2, 0, 0, 1, 1)

    def test_TensorForwardTest_CheckMaxPool2D_5x5x1_5x2(self):
        x_data = list(range(1, 5 * 5 + 1))
        y_data = [
            10,
            15,
            20,
            25,
        ]
        x_shape = Shape([5, 5])
        y_shape = Shape([1, 4])
        self.run_test_max_pool2d(x_shape, x_data, y_shape, y_data, 5, 2, 0, 0, 1, 1)

    def test_TensorForwardTest_CheckMaxPool2D_5x5x1_1x5(self):
        x_data = list(range(1, 5 * 5 + 1))
        y_data = [
            21, 22, 23, 24, 25,
        ]
        x_shape = Shape([5, 5])
        y_shape = Shape([5])
        self.run_test_max_pool2d(x_shape, x_data, y_shape, y_data, 1, 5, 0, 0, 1, 1)

    def test_TensorForwardTest_CheckMaxPool2D_5x5x1_2x5(self):
        x_data = list(range(1, 5 * 5 + 1))
        y_data = [
            22, 23, 24, 25,
        ]
        x_shape = Shape([5, 5])
        y_shape = Shape([4])
        self.run_test_max_pool2d(x_shape, x_data, y_shape, y_data, 2, 5, 0, 0, 1, 1)

    def test_TensorForwardTest_CheckMaxPool2D_5x5x1_5x5(self):
        x_data = list(range(1, 5 * 5 + 1))
        y_data = [25]
        x_shape = Shape([5, 5])
        y_shape = Shape([])
        self.run_test_max_pool2d(x_shape, x_data, y_shape, y_data, 5, 5, 0, 0, 1, 1)

    def test_TensorForwardTest_CheckMaxPool2D_5x5x3_2x2(self):
        x_data = list(range(1, 5 * 5 * 3 + 1))
        y_data = [
            # channel 1
             7,  8,  9, 10,
            12, 13, 14, 15,
            17, 18, 19, 20,
            22, 23, 24, 25,
            # channel 2
            32, 33, 34, 35,
            37, 38, 39, 40,
            42, 43, 44, 45,
            47, 48, 49, 50,
            # channel 3
            57, 58, 59, 60,
            62, 63, 64, 65,
            67, 68, 69, 70,
            72, 73, 74, 75,
        ]
        x_shape = Shape([5, 5, 3])
        y_shape = Shape([4, 4, 3])
        self.run_test_max_pool2d(x_shape, x_data, y_shape, y_data, 2, 2, 0, 0, 1, 1)

    def test_TensorForwardTest_CheckMaxPool2D_5x5x1_2x2_Padding10(self):
        x_data = list(range(1, 5 * 5 + 1))
        y_data = [
             6,  7,  8,  9, 10, 10,
            11, 12, 13, 14, 15, 15,
            16, 17, 18, 19, 20, 20,
            21, 22, 23, 24, 25, 25,
        ]
        x_shape = Shape([5, 5])
        y_shape = Shape([6, 4])
        self.run_test_max_pool2d(x_shape, x_data, y_shape, y_data, 2, 2, 1, 0, 1, 1)

    def test_TensorForwardTest_CheckMaxPool2D_5x5x1_2x2_Padding01(self):
        x_data = list(range(1, 5 * 5 + 1))
        y_data = [
             2,  3,  4,  5,
             7,  8,  9, 10,
            12, 13, 14, 15,
            17, 18, 19, 20,
            22, 23, 24, 25,
            22, 23, 24, 25,
        ]
        x_shape = Shape([5, 5])
        y_shape = Shape([4, 6])
        self.run_test_max_pool2d(x_shape, x_data, y_shape, y_data, 2, 2, 0, 1, 1, 1)

    def test_TensorForwardTest_CheckMaxPool2D_5x5x1_2x2_Padding11(self):
        x_data = list(range(1, 5 * 5 + 1))
        y_data = [
             1,  2,  3,  4,  5,  5,
             6,  7,  8,  9, 10, 10,
            11, 12, 13, 14, 15, 15,
            16, 17, 18, 19, 20, 20,
            21, 22, 23, 24, 25, 25,
            21, 22, 23, 24, 25, 25,
        ]
        x_shape = Shape([5, 5])
        y_shape = Shape([6, 6])
        self.run_test_max_pool2d(x_shape, x_data, y_shape, y_data, 2, 2, 1, 1, 1, 1)

    def test_TensorForwardTest_CheckMaxPool2D_5x5x1_2x2_Stride21(self):
        x_data = list(range(1, 5 * 5 + 1))
        y_data = [
             7,  9,
            12, 14,
            17, 19,
            22, 24,
        ]
        x_shape = Shape([5, 5])
        y_shape = Shape([2, 4])
        self.run_test_max_pool2d(x_shape, x_data, y_shape, y_data, 2, 2, 0, 0, 2, 1)

    def test_TensorForwardTest_CheckMaxPool2D_5x5x1_2x2_Stride12(self):
        x_data = list(range(1, 5 * 5 + 1))
        y_data = [
             7,  8,  9, 10,
            17, 18, 19, 20,
        ]
        x_shape = Shape([5, 5])
        y_shape = Shape([4, 2])
        self.run_test_max_pool2d(x_shape, x_data, y_shape, y_data, 2, 2, 0, 0, 1, 2)

    def test_TensorForwardTest_CheckMaxPool2D_5x5x1_2x2_Stride22(self):
        x_data = list(range(1, 5 * 5 + 1))
        y_data = [
             7,  9,
            17, 19,
        ]
        x_shape = Shape([5, 5])
        y_shape = Shape([2, 2])
        self.run_test_max_pool2d(x_shape, x_data, y_shape, y_data, 2, 2, 0, 0, 2, 2)

    def test_TensorForwardTest_CheckMaxPool2D_5x5x1_2x2_N(self):
        x_data = list(range(1, 5 * 5 * 3 + 1))
        y_data = [
            # minibatch 1
             7,  8,  9, 10,
            12, 13, 14, 15,
            17, 18, 19, 20,
            22, 23, 24, 25,
            # minibatch 2
            32, 33, 34, 35,
            37, 38, 39, 40,
            42, 43, 44, 45,
            47, 48, 49, 50,
            # minibatch 3
            57, 58, 59, 60,
            62, 63, 64, 65,
            67, 68, 69, 70,
            72, 73, 74, 75,
        ]
        x_shape = Shape([5, 5], 3)
        y_shape = Shape([4, 4], 3)
        self.run_test_max_pool2d(x_shape, x_data, y_shape, y_data, 2, 2, 0, 0, 1, 1)

    def test_TensorForwardTest_CheckMaxPool2D_VGG16ThirdLayer(self):
        # NOTE(odashi): 224*224*64 < 2^23 (float precision)
        x_data = list(range(1, 224 * 224 * 64 + 1))
        y_data = [0] * (112 * 112 * 64)
        for b in range(64):
            b_ofs = b * 224 * 224
            for x in range(112):
                x_ofs = b_ofs + (2 * x + 1) * 224
                for y in range(112):
                    y_data[y + b * 112 * 112 + x * 112] = x_ofs + 2 * y + 2
        x_shape = Shape([224, 224, 64])
        y_shape = Shape([112, 112, 64])
        self.run_test_max_pool2d(x_shape, x_data, y_shape, y_data, 2, 2, 0, 0, 2, 2)

    def test_TensorForwardTest_CheckInvalidPool2D(self):
        test_cases = [
            # invalid #dimensions
            (Shape([1, 1, 1, 2]), 1, 1, 0, 0, 1, 1, False),
            # zero-window/stride
            (Shape([]), 1, 1, 0, 0, 1, 1, True),
            (Shape([]), 0, 1, 0, 0, 1, 1, False),
            (Shape([]), 1, 0, 0, 0, 1, 1, False),
            (Shape([]), 1, 1, 0, 0, 0, 1, False),
            (Shape([]), 1, 1, 0, 0, 1, 0, False),
            # sizes mismatching
            (Shape([3, 3]), 3, 3, 0, 0, 1, 1, True),
            (Shape([3, 3]), 4, 3, 0, 0, 1, 1, False),
            (Shape([3, 3]), 3, 4, 0, 0, 1, 1, False),
            (Shape([3, 3]), 4, 4, 0, 0, 1, 1, False),
            # sizes mismatching with padding
            (Shape([3, 3]), 5, 5, 1, 1, 1, 1, True),
            (Shape([3, 3]), 6, 5, 1, 1, 1, 1, False),
            (Shape([3, 3]), 5, 6, 1, 1, 1, 1, False),
            (Shape([3, 3]), 6, 6, 1, 1, 1, 1, False),
            # sizes mismatching with stride
            (Shape([3, 3]), 3, 3, 0, 0, 2, 2, True),
            (Shape([3, 3]), 4, 3, 0, 0, 2, 2, False),
            (Shape([3, 3]), 3, 4, 0, 0, 2, 2, False),
            (Shape([3, 3]), 4, 4, 0, 0, 2, 2, False),
        ]
        for dev in TensorForwardTest.devices:
            for tc in test_cases:
                x = tF.constant(tc[0], 0, dev)
                if tc[7]:
                    try:
                        tF.max_pool2d(x, tc[1], tc[2], tc[3], tc[4], tc[5], tc[6])
                    except RuntimeError as e:
                        # TODO(vbkaisetsu):
                        # We have to implement a better method to detect
                        # NotImplementedError in Python
                        if "Not implemented" not in str(e):
                            raise
                else:
                    with self.assertRaises(RuntimeError) as e:
                        tF.max_pool2d(x, tc[1], tc[2], tc[3], tc[4], tc[5], tc[6])
