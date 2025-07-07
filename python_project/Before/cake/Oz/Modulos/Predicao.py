import numpy as np

class ModelPredictionHandler:
    def __init__(self, base_ponderacao=1.25):
        """
        Classe para lidar com predições de múltiplos modelos e ponderação dos resultados.
        
        Args:
            base_ponderacao (float): Base da exponencial usada para ponderar predições.
        """
        self.base = base_ponderacao

    def lista_predicao(self, i, t, modelos, array1):
        """
        Gera uma lista com possíveis predições para cada modelo.

        Args:
            i (int): Índice atual no processo.
            t (int): Quantidade de modelos contidos na lista original.
            modelos (list): Lista de modelos treinados.
            array1 (list): Lista de arrays (matrizes) de entrada para os modelos.

        Returns:
            list: Lista de predições feitas por cada modelo.
        """
        y_pred1 = []
        for sk in range(t):
            if modelos[sk] is not None:
                posicao = 60 * sk + 60
                print(f'Modelo {sk}, posição base: {posicao}')
                matriz1s = array1[sk]
                trick2 = i % 60
                order1 = 0 if trick2 == 59 else trick2 + 1

                x_new = np.array(matriz1s[order1, 3:])
                x_new = x_new.astype("float32").reshape(1, -1)
                print(f"Forma do input para o modelo: {x_new.shape}")

                predictions = modelos[sk].predict(x_new)
                y_pred = np.argmax(predictions) if predictions.ndim > 1 else predictions
                y_pred1.append(y_pred[0])
        
        print("Predições:", y_pred1)
        return y_pred1

    def ponderar_lista(self, lista):
        """
        Realiza ponderação dos elementos da lista com pesos exponenciais crescentes.

        Args:
            lista (list): Lista de inteiros (0 ou 1).

        Returns:
            int: Resultado ponderado final (0 ou 1).
        """
        n = len(lista)
        if n == 0:
            raise ValueError("A lista não pode estar vazia.")

        pesos = [self.base ** i for i in range(n)]
        soma_ponderada = sum(el * p for el, p in zip(lista, pesos))
        total_pesos = sum(pesos)

        qrange = 1 / n
        resultado = 1 if (soma_ponderada / total_pesos) >= qrange else 0

        print(f"Ponderação: {soma_ponderada:.4f} / {total_pesos:.4f} -> Resultado: {resultado}")
        return resultado
