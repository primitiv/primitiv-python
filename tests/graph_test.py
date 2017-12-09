import unittest

from primitiv import devices as D
from primitiv import functions as F
from primitiv import initializers as I
from primitiv import tensor_functions as tF
from primitiv import Device
from primitiv import Graph
from primitiv import Node
from primitiv import Parameter
from primitiv import Shape

import numpy as np


class GraphTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dev = D.Naive()
        cls.dev2 = D.Naive()

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_GraphTest_CheckInvalidNode(self):
        node = Node()
        self.assertFalse(node.valid())
        with self.assertRaises(RuntimeError):
            node.graph()
        with self.assertRaises(RuntimeError):
            node.operator_id()
        with self.assertRaises(RuntimeError):
            node.value_id()
        with self.assertRaises(RuntimeError):
            node.shape()
        with self.assertRaises(RuntimeError):
            node.device()
        with self.assertRaises(RuntimeError):
            node.to_float()
        with self.assertRaises(RuntimeError):
            node.to_list()
        with self.assertRaises(RuntimeError):
            node.to_ndarrays()
        with self.assertRaises(RuntimeError):
            node.backward()

    def test_GraphTest_CheckMultipleDevices(self):
        Device.set_default(GraphTest.dev)

        g = Graph()
        Graph.set_default(g)

        data1 = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
        data2 = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
        data3 = [1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6]
        grad = [12, 1]
        x1 = F.raw_input(Shape([2, 2], 3), data1)
        x2 = F.raw_input(Shape([2, 2], 3), data2, GraphTest.dev2)
        x3 = F.copy(x1, GraphTest.dev2) + x2
        self.assertEqual(Shape([2, 2], 3), x3.shape())
        self.assertIs(GraphTest.dev, x1.device())
        self.assertIs(GraphTest.dev2, x2.device())
        self.assertIs(GraphTest.dev2, x3.device())
        g.forward(x3)
        self.assertEqual(data1, g.forward(x1).to_list())
        self.assertEqual(data1, x1.to_list())
        self.assertEqual(data2, g.forward(x2).to_list())
        self.assertEqual(data2, x2.to_list())
        self.assertEqual(data3, g.forward(x3).to_list())
        self.assertEqual(data3, x3.to_list())

    def test_GraphTest_CheckInvalidMultipleDevices(self):
        Device.set_default(GraphTest.dev)

        g = Graph()
        Graph.set_default(g)

        dummy = [0] * 12
        x1 = F.raw_input(Shape([2, 2], 3), dummy)
        x2 = F.raw_input(Shape([2, 2], 3), dummy, GraphTest.dev2)
        x3 = x1 + x2
        with self.assertRaises(RuntimeError):
            g.forward(x3)

    def test_GraphTest_CheckClear(self):
        Device.set_default(GraphTest.dev)

        g = Graph()
        Graph.set_default(g)

        self.assertEqual(0, g.num_operators())

        F.raw_input([], [1])
        F.raw_input([], [1])
        self.assertEqual(2, g.num_operators())

        g.clear()
        self.assertEqual(0, g.num_operators())

        F.raw_input([], [1])
        F.raw_input([], [1])
        F.raw_input([], [1])
        self.assertEqual(3, g.num_operators())

        g.clear()
        self.assertEqual(0, g.num_operators())

        g.clear()
        self.assertEqual(0, g.num_operators())

    def test_GraphTest_CheckForwardBackward(self):
        Device.set_default(GraphTest.dev)

        g = Graph()
        Graph.set_default(g)

        data1 = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
        data3 = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

        nodes = []
        nodes.append(F.raw_input(Shape([2, 2], 3), data1))
        nodes.append(F.ones([2, 2]))
        nodes.append(F.raw_input(Shape([2, 2], 3), data3))
        nodes.append(nodes[0] + nodes[1])
        nodes.append(nodes[1] - nodes[2])
        nodes.append(nodes[3] * nodes[4])
        nodes.append(nodes[5] + 1)
        nodes.append(F.sum(nodes[6], 0))
        nodes.append(F.sum(nodes[7], 1))
        nodes.append(F.batch.sum(nodes[8]))

        self.assertEqual(10, len(nodes))
        self.assertEqual(10, g.num_operators())

        print(g.dump("dot"))

        expected_shapes = [
            Shape([2, 2], 3), Shape([2, 2]), Shape([2, 2], 3),
            Shape([2, 2], 3), Shape([2, 2], 3), Shape([2, 2], 3),
            Shape([2, 2], 3),
            Shape([1, 2], 3), Shape([], 3), Shape([]),
        ]
        for i, node in enumerate(nodes):
            self.assertEqual(expected_shapes[i], node.shape())
            self.assertIs(GraphTest.dev, node.device())

        g.forward(nodes[-1])

        expected_values = [
            [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            [1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
            [2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5],
            [1, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1],
            [2, 3, 4, 5, 0, 0, 0, 0, -2, -3, -4, -5],
            [3, 4, 5, 6, 1, 1, 1, 1, -1, -2, -3, -4],
            [7, 11, 2, 2, -3, -7],
            [18, 4, -10],
            [12],
        ]
        for i, node in enumerate(nodes):
            val = g.forward(node)
            self.assertTrue(val.valid())
            self.assertEqual(expected_values[i], val.to_list())
            self.assertEqual(expected_values[i], node.to_list())

    def test_GraphTest_CheckXor(self):
        Device.set_default(GraphTest.dev)

        w1 = Parameter([2, 2], I.Constant(0))
        w1.value += tF.raw_input([2, 2], [1, -1, 1, -1])
        b1 = Parameter([2], I.Constant(0))
        b1.value += tF.raw_input([2], [-1, -1])
        w2 = Parameter([1, 2], I.Constant(0))
        w2.value += tF.raw_input([1, 2], [1, 1])
        b2 = Parameter([], I.Constant(0))
        b2.value += tF.raw_input([], [1])

        inputs = [1, 1, 1, -1, -1, 1, -1, -1]
        outputs = [1, -1, -1, 1]

        g = Graph()
        Graph.set_default(g)

        nodes = []

        # sources
        nodes.append(F.raw_input(Shape([2], 4), inputs))
        nodes.append(F.parameter(w1))
        nodes.append(F.parameter(b1))
        nodes.append(F.parameter(w2))
        nodes.append(F.parameter(b2))
        # calculation
        nodes.append(F.matmul(nodes[1], nodes[0]));
        nodes.append(nodes[5] + nodes[2]);
        nodes.append(F.tanh(nodes[6]));
        nodes.append(F.matmul(nodes[3], nodes[7]));
        nodes.append(nodes[8] + nodes[4]);
        # losses
        nodes.append(F.raw_input(Shape([], 4), outputs));
        nodes.append(nodes[9] - nodes[10]);
        nodes.append(nodes[11] * nodes[11]);
        nodes.append(F.batch.sum(nodes[12]));

        self.assertEqual(len(nodes), g.num_operators())
        print(g.dump("dot"))

        g.forward(nodes[-1])

        # Check all node values.
        h1 = .76159416  # tanh(1)
        h2 = .99505475  # tanh(3)
        h3 = -.23346060 # tanh(1) - tanh(3)
        h4 = -1.5231883 # -2 * tanh(1)
        h5 = .76653940  # 1 + tanh(1) - tanh(3)
        h6 = -.52318831 # 1 - 2 * tanh(1)
        h7 = .47681169  # 2 - 2 * tanh(1)
        expected_values = [
            [1, 1, 1, -1, -1, 1, -1, -1],
            [1, -1, 1, -1],
            [-1, -1],
            [1, 1],
            [1],
            [2, -2, 0, 0, 0, 0, -2, 2],
            [1, -3, -1, -1, -1, -1, -3, 1],
            [h1, -h2, -h1, -h1, -h1, -h1, -h2, h1],
            [h3, h4, h4, h3],
            [h5, h6, h6, h5],
            [1, -1, -1, 1],
            [h3, h7, h7, h3],
            [h3 * h3, h7 * h7, h7 * h7, h3 * h3],
            [2 * (h3 * h3 + h7 * h7)],
        ]
        for i, node in enumerate(nodes):
            val = g.forward(nodes[i])
            self.assertTrue(val.valid())
            self.assertTrue(np.isclose(expected_values[i], val.to_list()).all())
            self.assertTrue(np.isclose(expected_values[i], node.to_list()).all())

    def test_GraphTest_CheckLSTM(self):
        Device.set_default(GraphTest.dev)

        pWix = Parameter([2, 2], I.Constant(0))
        pWix.value += tF.raw_input([2, 2], [.3, .1, .5, .3])
        pWfx = Parameter([2, 2], I.Constant(0))
        pWfx.value += tF.raw_input([2, 2], [.4, .1, .5, .8])
        pWox = Parameter([2, 2], I.Constant(0))
        pWox.value += tF.raw_input([2, 2], [.5, .9, .9, .7])
        pWjx = Parameter([2, 2], I.Constant(0))
        pWjx.value += tF.raw_input([2, 2], [.2, .6, .9, .3])
        pWih = Parameter([2, 2], I.Constant(0))
        pWih.value += tF.raw_input([2, 2], [.2, .3, .3, .3])
        pWfh = Parameter([2, 2], I.Constant(0))
        pWfh.value += tF.raw_input([2, 2], [.8, .4, .8, .3])
        pWoh = Parameter([2, 2], I.Constant(0))
        pWoh.value += tF.raw_input([2, 2], [.6, .2, .2, .7])
        pWjh = Parameter([2, 2], I.Constant(0))
        pWjh.value += tF.raw_input([2, 2], [.6, .4, .9, .5])
        pbi = Parameter([2], I.Constant(0))
        pbf = Parameter([2], I.Constant(0))
        pbo = Parameter([2], I.Constant(0))
        pbj = Parameter([2], I.Constant(0))

        g = Graph()
        Graph.set_default(g)

        x = F.raw_input(Shape([2], 2), [2, -2, 0.5, -0.5])
        h = F.raw_input(Shape([2], 2), [-1, 1, -0.5, 0.5])
        c = F.zeros([2])
        Wfx = F.parameter(pWfx)
        Wix = F.parameter(pWix)
        Wox = F.parameter(pWox)
        Wjx = F.parameter(pWjx)
        Wih = F.parameter(pWih)
        Wfh = F.parameter(pWfh)
        Woh = F.parameter(pWoh)
        Wjh = F.parameter(pWjh)
        bi = F.parameter(pbi)
        bf = F.parameter(pbf)
        bo = F.parameter(pbo)
        bj = F.parameter(pbj)

        i = F.sigmoid(F.matmul(Wix, x) + F.matmul(Wih, h) + bi)
        f = F.sigmoid(F.matmul(Wfx, x) + F.matmul(Wfh, h) + bf)
        o = F.sigmoid(F.matmul(Wox, x) + F.matmul(Woh, h) + bo)
        j = F.tanh(F.matmul(Wjx, x) + F.matmul(Wjh, h) + bj)
        cc = f * c + i * j
        hh = o * F.tanh(cc)

        t = F.zeros([2])
        diff = hh - t
        loss = diff * diff
        sum_loss = F.batch.sum(F.sum(loss, 0))

        self.assertEqual(45, g.num_operators());

        loss_tensor = g.forward(loss)
        sum_loss_tensor = g.forward(sum_loss)
        sum_loss.backward()

        expected_losses = [
            5.7667205e-03, 2.8605087e-02, 1.4819370e-03, 3.0073307e-03
        ]
        expected_sum_loss = sum(expected_losses)

        self.assertTrue(np.isclose(expected_losses, loss_tensor.to_list()).all())
        self.assertTrue(np.isclose(expected_losses, loss.to_list()).all())
        self.assertAlmostEqual(expected_sum_loss, sum_loss_tensor.to_float())
        self.assertAlmostEqual(expected_sum_loss, sum_loss.to_float())

        def print_node_val(name, value):
            print("%s: value=%s" % (name, value.to_ndarrays()))

        print("VALUES:")
        print_node_val("x", x)
        print_node_val("h", h)
        print_node_val("c", c)
        print_node_val("Wix", Wix)
        print_node_val("Wfx", Wfx)
        print_node_val("Wox", Wox)
        print_node_val("Wjx", Wjx)
        print_node_val("Wih", Wih)
        print_node_val("Wfh", Wfh)
        print_node_val("Woh", Woh)
        print_node_val("Wjh", Wjh)
        print_node_val("bi", bi)
        print_node_val("bf", bf)
        print_node_val("bo", bo)
        print_node_val("bj", bj)
        print_node_val("i", i)
        print_node_val("f", f)
        print_node_val("o", o)
        print_node_val("j", j)
        print_node_val("cc", cc)
        print_node_val("hh", hh)
        print_node_val("t", t)
        print_node_val("diff", diff)
        print_node_val("loss", loss)

    def test_GraphTest_CheckConcatLSTM(self):
        Device.set_default(GraphTest.dev)

        pWx = Parameter([8, 2], I.Constant(0))
        pWx.value += tF.raw_input([8, 2], [
            .3, .1, .4, .1, .5, .9, .2, .6,
            .5, .3, .5, .8, .9, .7, .9, .3,
        ])
        pWh = Parameter([8, 2], I.Constant(0))
        pWh.value += tF.raw_input([8, 2], [
            .2, .3, .8, .4, .6, .2, .6, .4,
            .3, .3, .8, .3, .2, .7, .9, .5,
        ])
        pb = Parameter([8], I.Constant(0))

        g = Graph()
        Graph.set_default(g)

        x = F.raw_input(Shape([2], 2), [2, -2, 0.5, -0.5])
        h = F.raw_input(Shape([2], 2), [-1, 1, -0.5, 0.5])
        c = F.zeros([2])
        Wx = F.parameter(pWx)
        Wh = F.parameter(pWh)
        b = F.parameter(pb)

        u = F.matmul(Wx, x) + F.matmul(Wh, h) + b
        i = F.sigmoid(F.slice(u, 0, 0, 2))
        f = F.sigmoid(F.slice(u, 0, 2, 4))
        o = F.sigmoid(F.slice(u, 0, 4, 6))
        j = F.tanh(F.slice(u, 0, 6, 8))
        cc = f * c + i * j
        hh = o * F.tanh(cc)

        t = F.zeros([2])
        diff = hh - t
        loss = diff * diff
        sum_loss = F.batch.sum(F.sum(loss, 0))

        self.assertEqual(28, g.num_operators())

        loss_tensor = g.forward(loss)
        sum_loss_tensor = g.forward(sum_loss)
        sum_loss.backward()

        expected_losses = [
            5.7667205e-03, 2.8605087e-02, 1.4819370e-03, 3.0073307e-03
        ]
        expected_sum_loss = sum(expected_losses)

        self.assertTrue(np.isclose(expected_losses, loss_tensor.to_list()).all())
        self.assertTrue(np.isclose(expected_losses, loss.to_list()).all());
        self.assertAlmostEqual(expected_sum_loss, sum_loss_tensor.to_float());
        self.assertAlmostEqual(expected_sum_loss, sum_loss.to_float());

        def print_node_val(name, value):
            print("%s: value=%s" % (name, value.to_ndarrays()))

        print("VALUES:")
        print_node_val("x", x)
        print_node_val("h", h)
        print_node_val("c", c)
        print_node_val("Wx", Wx)
        print_node_val("Wh", Wh)
        print_node_val("b", b)
        print_node_val("i", i)
        print_node_val("f", f)
        print_node_val("o", o)
        print_node_val("j", j)
        print_node_val("cc", cc)
        print_node_val("hh", hh)
        print_node_val("t", t)
        print_node_val("diff", diff)
        print_node_val("loss", loss)
