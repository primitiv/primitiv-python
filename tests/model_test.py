import tempfile
import unittest

from primitiv import Device
from primitiv import Model
from primitiv import Parameter
from primitiv import Shape
from primitiv.devices import Naive
from primitiv import initializers as I
from primitiv import tensor_functions as tF


class ModelTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = Naive()

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        Device.set_default(ModelTest.device)

    def tearDown(self):
        pass

    def test_ModelTest_CheckAddParameter(self):
        m = Model()
        p1 = Parameter()
        p2 = Parameter()
        p3 = Parameter()

        m.add("p1", p1)
        m.add("p1", p1)
        with self.assertRaises(RuntimeError):
            m.add("x", p1)

        with self.assertRaises(RuntimeError):
            m.add("p1", p2)
        m.add("p2", p2)
        m.add("p2", p2)
        with self.assertRaises(RuntimeError):
            m.add("x", p2)

        with self.assertRaises(RuntimeError):
            m.add("p1", p3)
        with self.assertRaises(RuntimeError):
            m.add("p2", p3)
        m.add("p3", p3)
        m.add("p3", p3)
        with self.assertRaises(RuntimeError):
            m.add("x", p3)

    def test_ModelTest_CheckAddSubmodel(self):
        m = Model()
        sm1 = Model()
        sm2 = Model()
        p1 = Parameter()
        p2 = Parameter()

        m.add("p1", p1)
        m.add("sm1", sm1)
        m.add("sm1", sm1)
        with self.assertRaises(RuntimeError):
            m.add("x", sm1)

        with self.assertRaises(RuntimeError):
            m.add("p1", p2)
        with self.assertRaises(RuntimeError):
            m.add("sm1", p2)
        with self.assertRaises(RuntimeError):
            m.add("p1", sm2)
        with self.assertRaises(RuntimeError):
            m.add("sm1", sm2)

        m.add("p2", p2)
        m.add("sm2", sm2)
        m.add("sm2", sm2)
        with self.assertRaises(RuntimeError):
            m.add("x", sm2)

    def test_ModelTest_CheckAddSubmodelCycle(self):
        m1 = Model()
        m2 = Model()
        m3 = Model()
        m4 = Model()

        with self.assertRaises(RuntimeError):
            m1.add("self", m1)

        m1.add("m2", m2)
        with self.assertRaises(RuntimeError):
            m2.add("m1", m1)

        m2.add("m3", m3)
        with self.assertRaises(RuntimeError):
            m3.add("m1", m1)
        with self.assertRaises(RuntimeError):
            m3.add("m2", m2)

        m2.add("m4", m4)
        with self.assertRaises(RuntimeError):
            m4.add("m1", m1)
        with self.assertRaises(RuntimeError):
            m4.add("m2", m2)

        m4.add("m3", m3)

    def test_ModelTest_CheckGetParameteer(self):
        m = Model()
        sm = Model()
        p1 = Parameter()
        p2 = Parameter()
        p3 = Parameter()
        m.add("p1", p1)
        m.add("p2", p2)
        sm.add("p3", p3)
        m.add("sm", sm)

        self.assertIs(p1, m["p1"])
        self.assertIs(p2, m["p2"])
        with self.assertRaises(TypeError):
            m["p3"]
        self.assertIs(sm, m["sm"])
        with self.assertRaises(TypeError):
            m["x"]

    def test_ModelTest_CheckGetParameterRecursiveByTuple(self):
        m = Model()
        sm = Model()
        p1 = Parameter()
        p2 = Parameter()
        p3 = Parameter()
        m.add("p1", p1)
        sm.add("p2", p2)
        sm.add("p3", p3)
        m.add("sm", sm)

        self.assertIs(p1, m["p1"])
        self.assertIs(p2, m["sm", "p2"])
        self.assertIs(p3, m["sm", "p3"])
        self.assertIs(p2, sm["p2"])
        self.assertIs(p3, sm["p3"])
        with self.assertRaises(TypeError):
            m["p2"]
        with self.assertRaises(TypeError):
            m["p3"]
        m["sm"]
        with self.assertRaises(TypeError):
            m["sm", "p1"]
        with self.assertRaises(TypeError):
            sm["p1"]
        with self.assertRaises(TypeError):
            m["x"]

    def test_ModelTest_CheckGetSubmodel(self):
        m = Model()
        sm1 = Model()
        sm2 = Model()
        ssm = Model()
        p = Parameter()
        m.add("p", p)
        m.add("sm1", sm1)
        m.add("sm2", sm2)
        sm1.add("ssm", ssm)

        self.assertIs(sm1, m["sm1"]);
        self.assertIs(sm2, m["sm2"]);
        with self.assertRaises(TypeError):
            m["ssm"]
        m["p"]

    def test_ModelTest_CheckGetSubmodelRecursiveByTuple(self):
        m = Model()
        sm1 = Model()
        sm2 = Model()
        ssm = Model()
        p = Parameter()
        m.add("p", p)
        m.add("sm1", sm1)
        m.add("sm2", sm2)
        sm1.add("ssm", ssm)

        self.assertIs(sm1, m["sm1"]);
        self.assertIs(sm2, m["sm2"]);
        self.assertIs(ssm, m["sm1", "ssm"]);
        self.assertIs(ssm, sm1["ssm"]);
        m["p"]
        with self.assertRaises(TypeError):
            m["ssm"]
        with self.assertRaises(TypeError):
            m["sm2", "ssm"]
        with self.assertRaises(TypeError):
            m["x"]

    def test_ModelTest_CheckGetAllParameters(self):
        m = Model()
        p1 = Parameter()
        p2 = Parameter()
        p3 = Parameter()
        m.add("p1", p1)
        m.add("p2", p2)
        m.add("p3", p3)
        params = m.get_all_parameters()
        self.assertEqual(3, len(params))
        self.assertIsInstance(params, dict)
        self.assertIs(p1, params[("p1",)])
        self.assertIs(p2, params[("p2",)])
        self.assertIs(p3, params[("p3",)])

    def test_ModelTest_CheckGetAllParametersWithSubmodels(self):
        m1 = Model()
        m2 = Model()
        m3 = Model()
        p1 = Parameter()
        p2 = Parameter()
        p3 = Parameter()
        m1.add("p", p1)
        m2.add("p", p2)
        m3.add("p", p3)
        m1.add("sm", m2)
        m2.add("sm", m3)

        params1 = m1.get_all_parameters()
        self.assertEqual(3, len(params1))
        self.assertIsInstance(params1, dict)
        self.assertIs(p1, params1[("p",)])
        self.assertIs(p2, params1[("sm", "p",)])
        self.assertIs(p3, params1[("sm", "sm", "p",)])

        params2 = m2.get_all_parameters()
        self.assertEqual(2, len(params2))
        self.assertIsInstance(params2, dict)
        self.assertIs(p2, params2[("p",)])
        self.assertIs(p3, params2[("sm", "p",)])

        params3 = m3.get_all_parameters()
        self.assertEqual(1, len(params3))
        self.assertIsInstance(params3, dict)
        self.assertIs(p3, params3[("p",)])

    def test_ModelTest_CheckGetTrainableParameters(self):
        m = Model()
        p1 = Parameter()
        p2 = Parameter()
        p3 = Parameter()
        m.add("p1", p1)
        m.add("p2", p2)
        m.add("p3", p3)
        params = m.get_trainable_parameters()
        self.assertEqual(3, len(params))
        self.assertIsInstance(params, dict)
        self.assertIs(p1, params[("p1",)]);
        self.assertIs(p2, params[("p2",)]);
        self.assertIs(p3, params[("p3",)]);

    def test_ModelTest_CheckGetTrainableParametersWithSubmodels(self):
        m1 = Model()
        m2 = Model()
        m3 = Model()
        p1 = Parameter()
        p2 = Parameter()
        p3 = Parameter()
        m1.add("p", p1)
        m2.add("p", p2)
        m3.add("p", p3)
        m1.add("sm", m2)
        m2.add("sm", m3)

        params1 = m1.get_trainable_parameters()
        self.assertEqual(3, len(params1))
        self.assertIsInstance(params1, dict)
        self.assertIs(p1, params1[("p",)])
        self.assertIs(p2, params1[("sm", "p",)])
        self.assertIs(p3, params1[("sm", "sm", "p",)])

        params2 = m2.get_trainable_parameters()
        self.assertEqual(2, len(params2))
        self.assertIsInstance(params2, dict)
        self.assertIs(p2, params2[("p",)])
        self.assertIs(p3, params2[("sm", "p",)])

        params3 = m3.get_trainable_parameters()
        self.assertEqual(1, len(params3))
        self.assertIsInstance(params3, dict)
        self.assertIs(p3, params3[("p",)])


    def test_ModelTest_CheckSaveLoad_Same(self):
        shape = Shape([2, 2])
        values1 = [1, 2, 3, 4]
        values2 = [5, 6, 7, 8]
        tmp = tempfile.NamedTemporaryFile()

        m1 = Model()
        m2 = Model()
        p1 = Parameter(shape, I.Constant(0))
        p1.value += tF.raw_input(shape, values1)
        p2 = Parameter(shape, I.Constant(0))
        p2.value += tF.raw_input(shape, values2)
        m1.add("p", p1)
        m2.add("p", p2)
        m1.add("sm", m2)

        m1.save(tmp.name)

        m1 = Model()
        m2 = Model()
        p1 = Parameter()
        p2 = Parameter()
        m1.add("p", p1)
        m2.add("p", p2)
        m1.add("sm", m2)

        m1.load(tmp.name)

        self.assertTrue(p1.valid())
        self.assertTrue(p2.valid())
        self.assertEqual(shape, p1.shape())
        self.assertEqual(shape, p2.shape())
        self.assertEqual(values1, p1.value.to_list())
        self.assertEqual(values2, p2.value.to_list())

    def test_ModelTest_CheckSaveLoad_Insufficient(self):
        shape = Shape([2, 2])
        values1 = [1, 2, 3, 4]
        values2 = [5, 6, 7, 8]
        tmp = tempfile.NamedTemporaryFile()

        m1 = Model()
        m2 = Model()
        p1 = Parameter(shape, I.Constant(0))
        p1.value += tF.raw_input(shape, values1)
        p2 = Parameter(shape, I.Constant(0))
        p2.value += tF.raw_input(shape, values2)
        m1.add("p", p1)
        m2.add("p", p2)
        m1.add("sm", m2)

        m1.save(tmp.name)

        m1 = Model()
        m2 = Model()
        p1 = Parameter()
        m1.add("p", p1)
        m1.add("sm", m2)

        with self.assertRaises(RuntimeError):
            m1.load(tmp.name)

    def test_ModelTest_CheckSaveLoad_Excessive(self):
        shape = Shape([2, 2])
        values1 = [1, 2, 3, 4]
        values2 = [5, 6, 7, 8]
        tmp = tempfile.NamedTemporaryFile()

        m1 = Model()
        m2 = Model()
        p1 = Parameter(shape, I.Constant(0))
        p1.value += tF.raw_input(shape, values1)
        p2 = Parameter(shape, I.Constant(0))
        p2.value += tF.raw_input(shape, values2)
        m1.add("p", p1)
        m2.add("p", p2)
        m1.add("sm", m2)

        m1.save(tmp.name)

        m1 = Model()
        m2 = Model()
        p1 = Parameter()
        p2 = Parameter()
        p3 = Parameter()
        m1.add("p", p1)
        m2.add("p", p2)
        m2.add("pp", p3)
        m1.add("sm", m2)

        m1.load(tmp.name)

        self.assertTrue(p1.valid())
        self.assertTrue(p2.valid())
        self.assertFalse(p3.valid())
        self.assertEqual(shape, p1.shape())
        self.assertEqual(shape, p2.shape())
        self.assertEqual(values1, p1.value.to_list())
        self.assertEqual(values2, p2.value.to_list())

    def test_ModelTest_CheckSaveLoadWithStats(self):
        shape = Shape([2, 2])
        values1 = [1, 2, 3, 4]
        values2 = [5, 6, 7, 8]
        stats1 = [10, 20, 30, 40]
        stats2 = [50, 60, 70, 80]
        tmp = tempfile.NamedTemporaryFile()

        m1 = Model()
        m2 = Model()
        p1 = Parameter(shape, I.Constant(0))
        p1.value += tF.raw_input(shape, values1)
        p2 = Parameter(shape, I.Constant(0))
        p2.value += tF.raw_input(shape, values2)
        p1.add_stats("a", shape)
        p2.add_stats("b", shape)
        p1.stats["a"].reset_by_vector(stats1);
        p2.stats["b"].reset_by_vector(stats2);
        m1.add("p", p1)
        m2.add("p", p2)
        m1.add("sm", m2)

        m1.save(tmp.name)

        m1 = Model()
        m2 = Model()
        p1 = Parameter()
        p2 = Parameter()
        m1.add("p", p1)
        m2.add("p", p2)
        m1.add("sm", m2)

        m1.load(tmp.name)

        self.assertTrue(p1.valid())
        self.assertTrue(p2.valid())
        self.assertEqual(shape, p1.shape())
        self.assertEqual(shape, p2.shape())
        self.assertEqual(values1, p1.value.to_list())
        self.assertEqual(values2, p2.value.to_list())
        self.assertTrue("a" in p1.stats)
        self.assertTrue("b" in p2.stats)
        self.assertEqual(stats1, p1.stats["a"].to_list())
        self.assertEqual(stats2, p2.stats["b"].to_list())

    def test_ModelTest_CheckSaveWithoutStats(self):
        shape = Shape([2, 2])
        values1 = [1, 2, 3, 4]
        values2 = [5, 6, 7, 8]
        stats1 = [10, 20, 30, 40]
        stats2 = [50, 60, 70, 80]
        tmp = tempfile.NamedTemporaryFile()

        m1 = Model()
        m2 = Model()
        p1 = Parameter(shape, I.Constant(0))
        p1.value += tF.raw_input(shape, values1)
        p2 = Parameter(shape, I.Constant(0))
        p2.value += tF.raw_input(shape, values2)
        p1.add_stats("a", shape)
        p2.add_stats("b", shape)
        p1.stats["a"].reset_by_vector(stats1)
        p2.stats["b"].reset_by_vector(stats2)
        m1.add("p", p1)
        m2.add("p", p2)
        m1.add("sm", m2)

        m1.save(tmp.name, False)

        m1 = Model()
        m2 = Model()
        p1 = Parameter()
        p2 = Parameter()
        m1.add("p", p1)
        m2.add("p", p2)
        m1.add("sm", m2)

        m1.load(tmp.name)

        self.assertTrue(p1.valid())
        self.assertTrue(p2.valid())
        self.assertEqual(shape, p1.shape())
        self.assertEqual(shape, p2.shape())
        self.assertEqual(values1, p1.value.to_list())
        self.assertEqual(values2, p2.value.to_list())
        self.assertFalse("a" in p1.stats)
        self.assertFalse("b" in p2.stats)

    def test_ModelTest_CheckLoadWithoutStats(self):
        shape = Shape([2, 2])
        values1 = [1, 2, 3, 4]
        values2 = [5, 6, 7, 8]
        stats1 = [10, 20, 30, 40]
        stats2 = [50, 60, 70, 80]
        tmp = tempfile.NamedTemporaryFile()

        m1 = Model()
        m2 = Model()
        p1 = Parameter(shape, I.Constant(0))
        p1.value += tF.raw_input(shape, values1)
        p2 = Parameter(shape, I.Constant(0))
        p2.value += tF.raw_input(shape, values2)
        p1.add_stats("a", shape)
        p2.add_stats("b", shape)
        p1.stats["a"].reset_by_vector(stats1)
        p2.stats["b"].reset_by_vector(stats2)
        m1.add("p", p1)
        m2.add("p", p2)
        m1.add("sm", m2)

        m1.save(tmp.name)

        m1 = Model()
        m2 = Model()
        p1 = Parameter()
        p2 = Parameter()
        m1.add("p", p1)
        m2.add("p", p2)
        m1.add("sm", m2)

        m1.load(tmp.name, False)

        self.assertTrue(p1.valid())
        self.assertTrue(p2.valid())
        self.assertEqual(shape, p1.shape())
        self.assertEqual(shape, p2.shape())
        self.assertEqual(values1, p1.value.to_list())
        self.assertEqual(values2, p2.value.to_list())
        self.assertFalse("a" in p1.stats)
        self.assertFalse("b" in p2.stats)
