from typing import List


class Odd:
    """
        Classe responsável pela coleta e classificação da odd.
    """

    def __init__(self, array_1: List[float], limite_clas: float = 2, limite: float = 3):
        """
        Inicializa a classe com um array e um limite opcional.

        Args:
            array_1 (List[float]): Lista de odds.
            limite_clas (float): Limite para classificação. Padrão: 2.0.
            limite (float): limite para os floats. Padrão: 3
        """
        if not all(isinstance(x, (int, float)) for x in array_1):
            raise TypeError("Todos os elementos do array_1 devem ser números.")
        self.array_1 = array_1
        self.limite_clas = limite_clas
        self.limite = limite

    def array_float(self) -> List[float]:
        """
        Substitui valores zero no array por 1.0.

        Returns:
            List[float]: Array modificado com zeros substituídos por 1.0.
        """
        return [1.0 if x == 0 else x for x in self.array_1]

    def array_int(self) -> List[int]:
        """
        Classifica as odds com base no limite.

        Returns:
            List[int]: Lista com valores 1 (abaixo do limite) ou 2
            (acima ou igual ao limite).
        """
        return [1 if x >= self.limite_clas else 0 for x in self.array_1]
    def array_truncado(self) -> List[int]:
        """
            Para entradas acima de 3, trunca 3.
            Returns:
                List[int]: Lista truncada.
        """
        return [3 if x >= self.limite else x for x in self.array_1]
