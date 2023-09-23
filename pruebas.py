import unittest
import numpy as np
import math
import Clasico_a_Cuantico


class Test_Clasico_a_Cuantico_functions(unittest.TestCase):

    def test_simula_canicas(self):
        # dinamica = [[0,0,0,0,0,0], [0,0,0,0,0,0], [0,1,0,0,0,1], [0,0,0,1,0,0],
        # [0,0,1,0,0,0],[1,0,0,0,1,0,]]
        # inicial = [6,2,1,5,3,10]
        # clicks = 2
        # resp = [0,0,9,5,12,1]
        dinamica = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 1, 0, ]]
        inicial = [6, 2, 1, 5, 3, 10]
        clicks = 2
        resultado = Clasico_a_Cuantico.simula_canicas(dinamica, inicial, clicks)
        resultado_esperado = [0, 0, 9, 5, 12, 1]
        self.assertTrue(np.allclose(resultado, resultado_esperado))

        # dinamica = [[0,0,0,0,0,0], [0,0,0,0,0,0], [0,1,0,0,0,1], [0,0,0,1,0,0],
        # [0,0,1,0,0,0],[1,0,0,0,1,0,]]
        # inicial = [6,2,1,5,3,10]
        # clicks = 1
        # resp = [0,0,12,5,1,9]
        dinamica = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 1, 0, ]]
        inicial = [6, 2, 1, 5, 3, 10]
        clicks = 1
        resultado = Clasico_a_Cuantico.simula_canicas(dinamica, inicial, clicks)
        resultado_esperado = [0, 0, 12, 5, 1, 9]
        self.assertTrue(np.allclose(resultado, resultado_esperado))

    def test_probabilistico(self):
        # dinamica = [[0, 0, 0, 0, 0, 0, 0, 0], [1/2, 0, 0, 0, 0, 0, 0, 0],
        #                      [1/2, 0, 0, 0, 0, 0, 0, 0], [0, 1/3, 0, 1, 0, 0, 0, 0],
        #                      [0, 1/3, 0, 0, 1, 0, 0, 0], [0, 1/3, 1/3, 0, 0, 1, 0, 0],
        #                      [0, 0, 1/3, 0, 0, 0, 1, 0], [0, 0, 1/3, 0, 0, 0, 0, 1]]
        # inicial = [1,0,0,0,0,0,0,0]
        # clicks = 1
        # resp = [0, 1/2, 1/2, 0, 0, 0, 0, 0]
        dinamica = [[0, 0, 0, 0, 0, 0, 0, 0], [1 / 2, 0, 0, 0, 0, 0, 0, 0],
                    [1 / 2, 0, 0, 0, 0, 0, 0, 0], [0, 1 / 3, 0, 1, 0, 0, 0, 0],
                    [0, 1 / 3, 0, 0, 1, 0, 0, 0], [0, 1 / 3, 1 / 3, 0, 0, 1, 0, 0],
                    [0, 0, 1 / 3, 0, 0, 0, 1, 0], [0, 0, 1 / 3, 0, 0, 0, 0, 1]]
        inicial = [1, 0, 0, 0, 0, 0, 0, 0]
        clicks = 1
        resultado = Clasico_a_Cuantico.probabilistico(dinamica, inicial, clicks)
        resultado_esperado = [0, 1 / 2, 1 / 2, 0, 0, 0, 0, 0]
        self.assertTrue(np.allclose(resultado, resultado_esperado))

        # dinamica = [[0, 0, 0, 0, 0, 0, 0, 0], [1/2, 0, 0, 0, 0, 0, 0, 0],
        #                      [1/2, 0, 0, 0, 0, 0, 0, 0], [0, 1/3, 0, 1, 0, 0, 0, 0],
        #                      [0, 1/3, 0, 0, 1, 0, 0, 0], [0, 1/3, 1/3, 0, 0, 1, 0, 0],
        #                      [0, 0, 1/3, 0, 0, 0, 1, 0], [0, 0, 1/3, 0, 0, 0, 0, 1]]
        # inicial = [1,0,0,0,0,0,0,0]
        # clicks = 1
        # resp = [0, 0, 0, 1/6, 1/6, 1/3, 1/6, 1/6]
        dinamica = [[0, 0, 0, 0, 0, 0, 0, 0], [1 / 2, 0, 0, 0, 0, 0, 0, 0],
                    [1 / 2, 0, 0, 0, 0, 0, 0, 0], [0, 1 / 3, 0, 1, 0, 0, 0, 0],
                    [0, 1 / 3, 0, 0, 1, 0, 0, 0], [0, 1 / 3, 1 / 3, 0, 0, 1, 0, 0],
                    [0, 0, 1 / 3, 0, 0, 0, 1, 0], [0, 0, 1 / 3, 0, 0, 0, 0, 1]]
        inicial = [1, 0, 0, 0, 0, 0, 0, 0]
        clicks = 2
        resultado = Clasico_a_Cuantico.probabilistico(dinamica, inicial, clicks)
        resultado_esperado = [0, 0, 0, 1 / 6, 1 / 6, 1 / 3, 1 / 6, 1 / 6]
        self.assertTrue(np.allclose(resultado, resultado_esperado))

    def test_cuantico(self):
        # dinamica = [[0, 1/math.sqrt(2), 1/math.sqrt(2), 0], [1/math.sqrt(2), 0, 0, -1/math.sqrt(2)],
        #                      [1/math.sqrt(2), 0, 0, 1/math.sqrt(2)],
        #                      [0, -1/math.sqrt(2), 1/math.sqrt(2), 0]]
        # inicial = [1,0,0,0]
        # clicks = 1
        # resp = [0, 1/math.sqrt(2), 1/math.sqrt(2), 0]
        dinamica = [[0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0], [1 / math.sqrt(2), 0, 0, -1 / math.sqrt(2)],
                    [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)],
                    [0, -1 / math.sqrt(2), 1 / math.sqrt(2), 0]]
        inicial = [1, 0, 0, 0]
        clicks = 1
        resultado = Clasico_a_Cuantico.cuantico(dinamica, inicial, clicks)
        resultado_esperado = [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0]
        self.assertTrue(np.allclose(resultado, resultado_esperado))

        # dinamica =[[0, 0, 0, 0, 0, 0, 0, 0], [1 / math.sqrt(2), 0, 0, 0, 0, 0, 0, 0],
        #                      [1 / math.sqrt(2), 0, 0, 0, 0, 0, 0, 0], [0, -1 + 1j / (math.sqrt(6)), 0, 1, 0, 0, 0, 0],
        #                      [0, -1 - 1j / (math.sqrt(6)), 0, 0, 1, 0, 0, 0],
        #                      [0, 1 - 1j / (math.sqrt(6)), -1 + 1j / (math.sqrt(6)), 0, 0, 1, 0, 0],
        #                      [0, 0, -1 - 1j / (math.sqrt(6)), 0, 0, 0, 1, 0],
        #                      [0, 0, 1 - 1j / (math.sqrt(6)), 0, 0, 0, 0, 1]]
        # inicial = [1,0,0,0,0,0,0,0]
        # clicks = 1
        # resp = [0, 1/math.sqrt(2), 1/math.sqrt(2),0,0,0,0,0]
        dinamica = [[0, 0, 0, 0, 0, 0, 0, 0], [1 / math.sqrt(2), 0, 0, 0, 0, 0, 0, 0],
                    [1 / math.sqrt(2), 0, 0, 0, 0, 0, 0, 0], [0, -1 + 1j / (math.sqrt(6)), 0, 1, 0, 0, 0, 0],
                    [0, -1 - 1j / (math.sqrt(6)), 0, 0, 1, 0, 0, 0],
                    [0, 1 - 1j / (math.sqrt(6)), -1 + 1j / (math.sqrt(6)), 0, 0, 1, 0, 0],
                    [0, 0, -1 - 1j / (math.sqrt(6)), 0, 0, 0, 1, 0], [0, 0, 1 - 1j / (math.sqrt(6)), 0, 0, 0, 0, 1]]
        inicial = [1,0,0,0,0,0,0,0]
        clicks = 1
        resultado = Clasico_a_Cuantico.cuantico(dinamica, inicial, clicks)
        resultado_esperado = [0, 1/math.sqrt(2), 1/math.sqrt(2),0,0,0,0,0]
        self.assertTrue(np.allclose(resultado, resultado_esperado))


if __name__ == '__main__':
    unittest.main()
