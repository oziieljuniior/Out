import numpy as np
from typing import Tuple, List, Union

class DataProcessor:
    """
    Classe para processamento de dados com operações em janelas deslizantes e transformação de matrizes.

    Atributos:
        window_size (int): Tamanho da janela deslizante (padrão: 59).
        click (int): Valor cumulativo para controle de transformação.
    """

    def __init__(self, window_size: int = 59, click: int = 60):
        """
        Inicializa o processador de dados.

        Args:
            window_size (int): Tamanho da janela deslizante (padrão: 59).
            click (int): Valor base para transformação de matriz (padrão: 60).
        """
        self.window_size = window_size
        self.click = click

    def calculate_means(self, data: Union[List[int], np.ndarray]) -> np.ndarray:
        """
        Calcula médias móveis em janelas deslizantes.

        Args:
            data: Array de inteiros (0 ou 1).

        Returns:
            np.ndarray: Médias móveis calculadas.
        """
        return np.array([
            np.mean(data[i-self.window_size:i]) 
            for i in range(self.window_size+1, len(data))
        ])

    def calculate_sums(self, data: Union[List[int], np.ndarray]) -> np.ndarray:
        """
        Calcula somas móveis em janelas deslizantes.

        Args:
            data: Array de inteiros (0 ou 1).

        Returns:
            np.ndarray: Somas móveis calculadas.
        """
        return np.array([
            sum(data[i-self.window_size:i]) 
            for i in range(self.window_size+1, len(data))
        ])

    def _reshape_to_matrix(self, data: np.ndarray) -> np.ndarray:
        """
        Redimensiona um array para matriz com linhas de tamanho 'click'.

        Args:
            data: Array a ser redimensionado.

        Returns:
            np.ndarray: Matriz resultante.
        """
        return data.reshape(-1, self.click)

    def _pad_data(self, data: np.ndarray) -> np.ndarray:
        """
        Preenche o array com dados aleatórios até atingir o tamanho adequado.

        Args:
            data: Array a ser preenchido.

        Returns:
            np.ndarray: Array preenchido.
        """
        while (len(data) - (self.window_size + 1)) % self.click != 0:
            new_data = np.random.choice([1, 0], size=self.click, p=[0.3, 0.7])
            data = np.concatenate((new_data, data))
        return data

    def transform_final_matrix(self, 
                            array_float: np.ndarray, 
                            array_int: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Transforma os arrays de entrada na matriz final.

        Args:
            array_float: Array com entradas vetorizadas float.
            array_int: Array com entradas vetorizadas int.

        Returns:
            Tuple contendo:
                - Matriz de features (float)
                - Matriz de labels (int)
                - Posição de referência
        """
        # Verificação e preenchimento dos dados
        if len(array_int) < self.window_size + 2:
            raise ValueError(f"Array int deve ter pelo menos {self.window_size + 2} elementos")
        
        array_int = self._pad_data(array_int)

        # Cálculos das métricas
        moving_sums = self.calculate_sums(array_int)[1:]
        moving_avgs = self.calculate_means(array_int)[1:]

        # Transformação em matrizes
        sum_matrix = self._reshape_to_matrix(moving_sums)
        avg_matrix = self._reshape_to_matrix(moving_avgs)
        float_matrix = self._reshape_to_matrix(array_float[1:])
        int_matrix = self._reshape_to_matrix(array_int[1:])

        # Remoção da primeira coluna (se necessário)
        float_matrix = float_matrix[:, 1:]
        int_matrix = int_matrix[:, 1:]

        # Empilhamento das features
        stacked_features = np.stack([float_matrix, sum_matrix, avg_matrix], axis=2)
        final_features = stacked_features.reshape(stacked_features.shape[0], -1)

        # Cálculo da posição de referência
        ref_position = (self.click // self.window_size) - 1

        return final_features, int_matrix, ref_position

# Exemplo de uso
if __name__ == "__main__":
    # Dados de exemplo
    np.random.seed(42)
    float_data = np.random.rand(300)  # 300 valores float
    int_data = np.random.randint(0, 2, 30)  # 300 valores 0 ou 1

    # Processamento
    processor = DataProcessor(window_size=59, click=60)
    
    try:
        features, labels, ref_pos = processor.transform_final_matrix(float_data, int_data)
        print("Matriz de features shape:", features.shape)
        print("Matriz de labels shape:", labels.shape)
        print("Posição de referência:", ref_pos)
    except ValueError as e:
        print(f"Erro: {e}")