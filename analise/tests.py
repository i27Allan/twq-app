from django.test import TestCase
import numpy as np
from analise import icc
from analise import pearson_spearman as ps
from analise import shapiro_wilk as sw
from analise import plot_data_normal_graph as pdng

class AnaliseTest(TestCase):

    def setUp(self):
        self.data = np.array([
            [9,2,5,8],
            [6,1,3,2],
            [8,4,6,8],
            [7,1,2,6],
            [10,5,6,9],
            [6,2,4,7]
        ])

    def test_icc(self):
        resultado_esperado = 0.289763779528 #para o icc2_1
        self.assertAlmostEqual(resultado_esperado, icc.icc2_1(self.data))
        resultado_esperado = 0.607528065155 #para o icc2_k
        self.assertAlmostEqual(resultado_esperado, icc.icc2_k(self.data))
        resultado_esperado = 0.714840714841  #para o icc3_1
        self.assertAlmostEqual(resultado_esperado, icc.icc3_1(self.data))
        resultado_esperado = 0.909315542377  #para o icc3_k
        self.assertAlmostEqual(resultado_esperado, icc.icc3_k(self.data))

    def test_pearson(self):
        resultado_esperado = [
            [1, 0.74535599, 0.725, 0.75017728],
            [0.74535599, 1, 0.89442719, 0.72932496],
            [0.725, 0.89442719, 1, 0.71756088],
            [0.75017728, 0.72932496, 0.71756088, 1, ]
        ]
        resultado_obtido = ps.pearson(self.data)
        self.assertEqual(len(resultado_esperado), len(resultado_obtido))

        for i in range(0, len(resultado_esperado)):
            self.assertEqual(len(resultado_esperado[i]), len(resultado_obtido[i]))
            for j in range(0, len(resultado_esperado[i])):
                self.assertAlmostEqual(resultado_esperado[i][j], resultado_obtido[i][j])

    def test_spearman(self):
        resultado_esperado = [
            [1, 0.71649772, 0.70588235, 0.88235294],
            [0.71649772, 1, 0.95533029, 0.94040326],
            [0.70588235, 0.95533029, 1, 0.89705882],
            [0.88235294, 0.94040326, 0.89705882, 1]
        ]
        resultado_obtido = ps.spearman(self.data)
        self.assertEqual(len(resultado_esperado), len(resultado_obtido))

        for i in range(0, len(resultado_esperado)):
            self.assertEqual(len(resultado_esperado[i]), len(resultado_obtido[i]))
            for j in range(0, len(resultado_esperado[i])):
                self.assertAlmostEqual(resultado_esperado[i][j], resultado_obtido[i][j])

    def test_shapiro_wilk(self):
        w_esperado = 0.9439817070960999
        p_value_esperado = 0.19998089969158173

        w_obtido, p_value_obtido = sw.shapiro_wilk(self.data)
        self.assertAlmostEqual(w_esperado, w_obtido)
        self.assertAlmostEqual(p_value_esperado, p_value_obtido)


                

    