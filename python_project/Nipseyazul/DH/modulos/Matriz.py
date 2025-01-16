from typing import List
import numpy as np


class Ordernar60:
    """
        Classe responsavel pela ordenação do array em matriz.
    """

    def __init__(self, array1: List[float]):
        """
            Inicio com o array e tamanho dele.
        """
        self.array1 = array1
        self.tamanho = len(array1)

    def ordernar_colunas(self) -> List[int]:
        """
            Lista os tipos de colunas que podem ser tranformadas
            em matrizes.
            Returns:
                List[int]: Lista com tipos de matrizes.
        """
        tamanho = self.tamanho // 5
        lista = [name for name in range(60, tamanho, 60)]

        info = []
        for name in lista:
            order = tamanho % name
            if order == 0:
                info.append(name)
        return info

    def tranformar(self) -> List[np.array]:
        """
            Transformar array em matriz com uma lista determinada.
            returns:
                List[np.array, ..., np.array]: Uma lista com matrizes.
        """
        info = Ordernar60(self.array1).ordernar_colunas()
        print(f'Novas matrizes para: {info} ...')
        final = []
        for name in info:
            order1 = self.tamanho // name
            print(name, order1, self.tamanho)
            if order1 >= 5:
                matriz = np.array(self.array1).reshape(-1, name).T
                final.append(matriz)
        return final
