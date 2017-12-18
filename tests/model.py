from primitiv import Parameter, Device, Model, Shape
from primitiv import devices as D
from primitiv import initializers as I
from primitiv import tensor_functions as tF

import numpy as np

import unittest
import tempfile


class TestModel(Model):
    # NOTE(vbkaisetsu):
    # Custom models can be created without calling super().__init__()
    # function.
    # This override suppresses calling __init__() function of
    # the parent Model class to simulate the actual model implementation.
    def __init__(self):
        pass


class ModelTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.dev = D.Naive()
        Device.set_default(self.dev)

    def tearDown(self):
         pass

    def test_model_load_save(self):
        submodel = TestModel()
        sp1 = Parameter([2, 4], I.Constant(0))
        sp1.value = tF.input(np.array([[0,1,2,3],[4,5,6,7]]))
        sp2 = Parameter([2, 4], I.Constant(0))
        sp2.value = tF.input(np.array([[9,8,7,6],[5,4,3,2]]))
        submodel.add("sp1", sp1)
        submodel.add("sp2", sp2)
        parentmodel = TestModel()
        p1 = Parameter([4, 2], I.Constant(0))
        p1.value = tF.input(np.array([[0,1],[2,3],[4,5],[6,7]]))
        p2 = Parameter([4, 2], I.Constant(0))
        p2.value = tF.input(np.array([[9,8],[7,6],[5,4],[3,2]]))
        parentmodel.add("p1", p1)
        parentmodel.add("p2", p2)
        parentmodel.add("sub", submodel)
        submodel_load = TestModel()
        sp1 = Parameter()
        sp2 = Parameter()
        submodel_load.add("sp1", sp1)
        submodel_load.add("sp2", sp2)
        parentmodel_load = TestModel()
        p1 = Parameter()
        p2 = Parameter()
        parentmodel_load.add("p1", p1)
        parentmodel_load.add("p2", p2)
        parentmodel_load.add("sub", submodel_load)
        with tempfile.NamedTemporaryFile() as fp:
            parentmodel.save(fp.name)
            parentmodel_load.load(fp.name)
        self.assertTrue((parentmodel_load["p1"].value.to_ndarrays()[0] == np.array([[0,1],[2,3],[4,5],[6,7]])).all())
        self.assertTrue((parentmodel_load["p2"].value.to_ndarrays()[0] == np.array([[9,8],[7,6],[5,4],[3,2]])).all())
        self.assertTrue((parentmodel_load["sub", "sp1"].value.to_ndarrays()[0] == np.array([[0,1,2,3],[4,5,6,7]])).all())
        self.assertTrue((parentmodel_load["sub", "sp2"].value.to_ndarrays()[0] == np.array([[9,8,7,6],[5,4,3,2]])).all())

    def test_model_parameter(self):
        model = Model()
        param = Parameter()
        model.add("p", param)
        self.assertIs(model["p"], param)
        self.assertIs(model[("p",)], param)

    def test_model_parameter_deep(self):
        model1 = Model()
        model2 = Model()
        model1.add("m2", model2)
        model3 = Model()
        model2.add("m3", model3)
        param = Parameter()
        model3.add("p", param)
        self.assertIs(model1["m2", "m3", "p"], param)
        self.assertIs(model1["m2"]["m3"]["p"], param)

    def test_model_submodel(self):
        model1 = Model()
        model2 = Model()
        model1.add("m", model2)
        self.assertIs(model1["m"], model2)

    def test_model_submodel_deep(self):
        model1 = Model()
        model2 = Model()
        model1.add("m2", model2)
        model3 = Model()
        model2.add("m3", model3)
        model4 = Model()
        model3.add("m4", model4)
        self.assertIs(model1["m2", "m3", "m4"], model4)
        self.assertIs(model1["m2"]["m3"]["m4"], model4)

    def test_model_invalid_operation(self):
        model1 = Model()
        model2 = Model()
        model1.add("m", model2)
        param = Parameter()
        model1.add("p", param)
        with self.assertRaises(TypeError) as e:
            model1["notfound"]
        self.assertEqual(str(e.exception), "'name' is not a name of neither parameter nor submodel")
        with self.assertRaises(TypeError):
            del model1["p"]
        with self.assertRaises(TypeError):
            del model1["m"]
        with self.assertRaises(TypeError):
            del model1[0]
        with self.assertRaises(TypeError):
            model1[(0, 1)]
        with self.assertRaises(TypeError):
            model1[[0, 1]]
        model3 = TestModel()
        model3.p = Parameter()
        model3.m = TestModel()
        model3.a = "test"
        del model3.a
        self.assertNotIn("a", model3.__dict__)
        with self.assertRaises(TypeError):
            del model3.p
        self.assertIn("p", model3.__dict__)
        with self.assertRaises(TypeError):
            del model3.m
        self.assertIn("m", model3.__dict__)

    def test_model_get_all_parameters(self):
        submodel = TestModel()
        sp1 = Parameter()
        sp2 = Parameter()
        submodel.add("sp1", sp1)
        submodel.add("sp2", sp2)
        parentmodel = TestModel()
        p1 = Parameter()
        p2 = Parameter()
        sub = submodel
        parentmodel.add("p1", p1)
        parentmodel.add("p2", p2)
        parentmodel.add("sub", sub)
        params = parentmodel.get_all_parameters()
        self.assertIs(params[("p1",)], p1)
        self.assertIs(params[("p2",)], p2)
        self.assertIs(params[("sub", "sp1")], sp1)
        self.assertIs(params[("sub", "sp2")], sp2)

    def test_model_setattr(self):
        model = TestModel()
        model.p1 = Parameter()
        model.p2 = Parameter()
        model.p3 = Parameter()
        model.m1 = TestModel()
        model.m2 = TestModel()
        model.m3 = TestModel()
        self.assertIs(model["p1"], model.p1)
        self.assertIs(model["p2"], model.p2)
        self.assertIs(model["p3"], model.p3)
        model.p4 = Parameter()
        self.assertIs(model["m1"], model.m1)
        self.assertIs(model["m2"], model.m2)
        self.assertIs(model["m3"], model.m3)
        model.m4 = TestModel()
        self.assertIs(model["p4"], model.p4)
        self.assertIs(model["m4"], model.m4)
