import numpy as np
import pandas as pd

class Placar:
    def __init__(self, offset=6000):
        self.offset = offset
        self.array_geral = {
            'Contador_Geral': 0,
            'Contador_Sintetico': 0,
            'Precisao_Geral': 0.0,
            'i_sintetico': 0,
            'Precisao_Sintetica': 0.0
        }
    
    def atualizar_geral(self, i: int, resultado: int, odd: float) -> dict:
        i = i - self.offset  # Ajuste para deslocamento temporal
        if resultado == 1:
            self.array_geral['i_sintetico'] += 1
            if odd >= 3:
                self.array_geral['Contador_Geral'] += 1
                self.array_geral['Contador_Sintetico'] += 1
        else:
            if odd >= 3:
                self.array_geral['Contador_Geral'] += 1

        # Atualiza métricas
        self.array_geral['Precisao_Geral'] = (
            (self.array_geral['Contador_Geral'] / i) * 100 if i > 0 else 0.0
        )
        self.array_geral['Precisao_Sintetica'] = (
            (self.array_geral['Contador_Sintetico'] / self.array_geral['i_sintetico']) * 100
            if self.array_geral['i_sintetico'] > 0 else 0.0
        )

        return self.array_geral

class Placar60:
    def __init__(self):
        self.df1 = pd.DataFrame({'lautgh1': np.zeros(60, dtype=int), 'lautgh2': np.zeros(60, dtype=int)})
        self.media_parray = []

    def atualizar(self, i: int, resultado: int, odd: float) -> list:
        core1 = i % 60
        if resultado == 1:
            if odd >= 3:
                self.df1.iloc[core1, :] += 1
            else:
                self.df1.iloc[core1, 1] += 1
            medida_pontual = self.df1.iloc[core1, 0] / self.df1.iloc[core1, 1]
        else:
            if len(self.media_parray) < 59:
                medida_pontual = 0
            else:
                medida_pontual = self.media_parray[-60:]

        self.media_parray.append(medida_pontual)
        return self.media_parray
