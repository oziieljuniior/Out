from typing import List


class Odd:
    """
        Classe responsável pela coleta e classificação da odd.
    """

    def __init__(self, array_1: List[float], limite: float = 2):
        """
        Inicializa a classe com um array e um limite opcional.

        Args:
            array_1 (List[float]): Lista de odds.
            limite (float): Limite para classificação. Padrão: 2.0.
        """
        self.array_1 = array_1
        self.limite = limite

    def array_float(self) -> List[float]:
        """
        Substitui valores zero no array por 1.0.

        Returns:
            List[float]: Array modificado com zeros substituídos por 1.0.
        """
        return [1.0 if x == 0 else x for x in self.array1]

    def array_int(self) -> List[int]:
        """
        Classifica as odds com base no limite.

        Returns:
            List[int]: Lista com valores 1 (abaixo do limite) ou 2
            (acima ou igual ao limite).
        """
        return [1 if x >= self.limite else 0 for x in self.array1]
