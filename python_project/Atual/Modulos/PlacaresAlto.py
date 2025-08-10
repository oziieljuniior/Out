import numpy as np

class Placar:
    def __init__(self):
        self.array_geral = {
            'Contador_Geral': 0,
            'Contador_Sintetico': 0,
            'Precisao_Geral': 0.0,
            'i_sintetico': 0,
            'Precisao_Sintetica': 0.0
        }
    
    def atualizar_geral(self, i: int, resultado: int, odd: float) -> dict:
        i = i - 12000  # Ajuste para deslocamento temporal
        if resultado == 1:
            self.array_geral['i_sintetico'] += 1
            if odd >= 30:
                self.array_geral['Contador_Geral'] += 1
                self.array_geral['Contador_Sintetico'] += 1
        else:
            if odd >= 30:
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
