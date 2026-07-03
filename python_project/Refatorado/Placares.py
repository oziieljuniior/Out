import numpy as np
import pandas as pd

class Placar:
    """
    Classe para monitoramento e contagem geral de acertos das predições de Odds.
    """
    def __init__(self, offset: int = 6000):
        self.offset = offset
        self.array_geral = {
            'Contador_Geral': 0,
            'Contador_Sintetico': 0,
            'Precisao_Geral': 0.0,
            'i_sintetico': 0,
            'Precisao_Sintetica': 0.0
        }
    
    def atualizar_geral(self, i: int, resultado: int, odd: float) -> dict:
        """
        Atualiza o painel geral de contagens e precisão acumulada.
        """
        i_ajustado = i - self.offset  # Ajuste para deslocamento temporal
        
        if resultado == 1:
            self.array_geral['i_sintetico'] += 1
            if odd >= 3:
                self.array_geral['Contador_Geral'] += 1
                self.array_geral['Contador_Sintetico'] += 1
        else:
            if odd >= 3:
                self.array_geral['Contador_Geral'] += 1

        # Atualiza métricas com proteção contra divisão por zero
        self.array_geral['Precisao_Geral'] = (
            (self.array_geral['Contador_Geral'] / i_ajustado) * 100 if i_ajustado > 0 else 0.0
        )
        self.array_geral['Precisao_Sintetica'] = (
            (self.array_geral['Contador_Sintetico'] / self.array_geral['i_sintetico']) * 100
            if self.array_geral['i_sintetico'] > 0 else 0.0
        )

        return self.array_geral

class Placar60:
    """
    Classe para monitoramento de acertos baseado em 60 sub-modelos correspondentes
    a cada passo do ciclo temporal (i % 60).
    """
    def __init__(self):
        # lautgh1: acertos (odds >= 3), lautgh2: total de palpites sintéticos realizados
        self.df1 = pd.DataFrame({
            'lautgh1': np.zeros(60, dtype=int), 
            'lautgh2': np.zeros(60, dtype=int)
        })
        self.media_parray = []

    def atualizar(self, i: int, resultado: int, odd: float) -> list:
        """
        Atualiza a acurácia para o modelo correspondente ao resto (i % 60).
        """
        core1 = i % 60
        if resultado == 1:
            if odd >= 3:
                self.df1.iloc[core1, :] += 1
            else:
                self.df1.iloc[core1, 1] += 1
            
            total_palpites = self.df1.iloc[core1, 1]
            medida_pontual = (self.df1.iloc[core1, 0] / total_palpites) if total_palpites > 0 else 0.0
        else:
            # Bug fix: Modificado de fatiamento [-60:] para o índice específico [-60] 
            # para evitar listas aninhadas em media_parray e subsequente TypeError
            if len(self.media_parray) < 60:
                medida_pontual = 0.0
            else:
                medida_pontual = self.media_parray[-60]

        self.media_parray.append(medida_pontual)
        return self.media_parray
