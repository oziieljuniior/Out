import numpy as np
import pandas as pd

class ScoreManager:
    def __init__(self):
        """
        Inicializa o gerenciador de placar. Você pode expandir aqui se quiser incluir
        estado interno (ex: df1, array_geral etc).
        """
        pass

    def placar60(self, df1, i, media_parray, resultado, odd):
        """
        Atualiza o placar das últimas 60 entradas com base no resultado e na odd.

        Args:
            df1 (pd.DataFrame): DataFrame com estatísticas por posição.
            i (int): Índice atual (posição da entrada).
            media_parray (list): Lista com histórico de acurácia por posição.
            resultado (int): Resultado atual (0 ou 1).
            odd (float): Odd usada na entrada.

        Returns:
            list: Lista atualizada com a acurácia pontual da entrada.
        """
        core1 = i % 60
        if resultado == 1:
            if odd >= 4:
                df1.iloc[core1, :] += 1
                medida_pontual = df1.iloc[core1, 0] / df1.iloc[core1, 1]
            else:
                df1.iloc[core1, 1] += 1
                medida_pontual = df1.iloc[core1, 0] / df1.iloc[core1, 1]
        else:
            if len(media_parray) < 59:
                medida_pontual = 0
            else:
                medida_pontual = media_parray[-60]

        media_parray.append(medida_pontual)
        return media_parray

    def placargeral(self, resultado, odd, array_geral):
        """
        Atualiza estatísticas globais de acurácia e precisão.

        Args:
            resultado (int): Resultado da predição ponderada (0 ou 1).
            odd (float): Odd utilizada.
            array_geral (list): Lista com estatísticas na ordem:
                [acurácia, precisão, acerto, acerto1, j, j1]

        Returns:
            list: Lista atualizada com as estatísticas globais.
        """
        if resultado == 1:
            if odd >= 4:
                array_geral[2] += 1  # acerto
                array_geral[3] += 1  # acerto1
                array_geral[4] += 1  # j
                array_geral[5] += 1  # j1
            else:
                array_geral[4] += 1
                array_geral[5] += 1
        else:
            if odd < 4:
                array_geral[3] += 1  # acerto1
                array_geral[5] += 1  # j1
            else:
                array_geral[5] += 1

        # Calcular acurácia e precisão (%)
        if array_geral[4] != 0 and array_geral[5] != 0:
            array_geral[0] = (array_geral[2] / array_geral[4]) * 100
            array_geral[1] = (array_geral[3] / array_geral[5]) * 100
        else:
            array_geral[0], array_geral[1] = 0, 0

        return array_geral
