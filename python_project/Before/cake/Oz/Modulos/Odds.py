import numpy as np
import skfuzzy as fuzz
import pandas as pd

class FuzzyOddsProcessor:
    def __init__(self, intervalo_min=1, intervalo_max=8):
        self.odd_range = np.arange(intervalo_min, intervalo_max + 0.1, 0.1)
        
        # Conjuntos fuzzy
        self.muito_baixo = fuzz.trimf(self.odd_range, [1, 1, 2.5])
        self.baixo = fuzz.trimf(self.odd_range, [1.5, 3, 4.5])
        self.medio = fuzz.trimf(self.odd_range, [3.5, 5, 6.5])
        self.alto = fuzz.trimf(self.odd_range, [5.5, 7, 8])
        self.muito_alto = fuzz.trimf(self.odd_range, [7, 8, 8])

    def fuzzy_classification(self, odd):
        """
        Classifica a odd com base em lógica fuzzy, retornando nível de confiança.
        """
        # Graus de pertinência
        pert_muito_baixo = fuzz.interp_membership(self.odd_range, self.muito_baixo, odd)
        pert_baixo = fuzz.interp_membership(self.odd_range, self.baixo, odd)
        pert_medio = fuzz.interp_membership(self.odd_range, self.medio, odd)
        pert_alto = fuzz.interp_membership(self.odd_range, self.alto, odd)
        pert_muito_alto = fuzz.interp_membership(self.odd_range, self.muito_alto, odd)

        max_pert = max(pert_muito_baixo, pert_baixo, pert_medio, pert_alto, pert_muito_alto)

        if max_pert == 0:
            return 0.0
        if max_pert == pert_muito_alto:
            return 1.0
        elif max_pert == pert_alto:
            return 0.8
        elif max_pert == pert_medio:
            return 0.6
        elif max_pert == pert_baixo:
            return 0.4
        elif max_pert == pert_muito_baixo:
            return 0.2
        else:
            return 0.0

    def coletar_odd(self, i, inteiro, data, array2s, array2n, alavanca=True):
        """
        Coleta e processa uma entrada (odd) com lógica fuzzy e classificação binária.
        """
        if i <= inteiro:
            if alavanca:
                odd = float(data['Entrada'][i].replace(",", '.'))
            else:
                odd = data['Entrada'][i]
            if odd == 0:
                odd = 1
            print(f'Entrada: {odd}')
        else:
            odd = float(input("Entrada -> ").replace(",", '.'))

        if odd == 0:
            return array2s, array2n, odd
        if odd >= 8:
            odd = 8

        # Aplicando lógica fuzzy
        corte1 = self.fuzzy_classification(odd)
        array2s.append(corte1)

        # Classificação binária simples
        corte2 = 1 if odd >= 4 else 0
        array2n.append(corte2)

        return array2s, array2n, odd
