from typing import List
import numpy as np


class Ordernar:
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
            Ordena os tipos de colunas que podem ser tranformadas
            em matrizes, sempre multiplo de 60.
            Returns:
                List[int]: Lista com tipos de matrizes.
        """
        lista = [name for name in range(60, self.tamanho, 60)]

        info = []
        #print(lista)
        for name in lista:
            order = self.tamanho % name
            limite = self.tamanho // name
            if order == 0 and limite >= 5:
                info.append(name)
        return info

    def tranformar(self) -> List[np.array]:
        """
            Transformar array em matriz com uma lista determinada.
            returns:
                List[np.array, ..., np.array]: Uma lista com matrizes.
        """
        info = Ordernar(self.array1).ordernar_colunas()
        print(f'Novas matrizes para: {info} ...')
        final = []
        for name in info:
            order1 = self.tamanho // name
            print(name, order1, self.tamanho)
            if order1 >= 5:
                matriz = np.array(self.array1).reshape(-1, name).T
                final.append(matriz)
        return final
