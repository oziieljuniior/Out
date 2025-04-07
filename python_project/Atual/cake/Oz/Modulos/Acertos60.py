import pandas as pd
import numpy as np

class OperacoesMatriz():
    def __init__(self):
        pass
    
    def calculate_orders(array4):
        """
        Calcula a soma dos elementos de array4 em janelas deslizantes de 59 elementos.

        Args:
            array4 (list): Lista de inteiros (0 ou 1).

        Returns:
            list: Lista com a soma dos elementos em janelas de 59 elementos.
        """
        array5 = []
        for i in range(len(array4) - 1):
            if i >= 59:
                order = sum(array4[i - 59: i])
                array5.append(order)
        return array5

    # Exemplo de uso:
    # array4 = [0, 1, 0, 1, ...]  # Supondo que array4 tenha elementos suficientes
    # array5 = calculate_orders(array4)

