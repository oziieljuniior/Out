import numpy as np

class MatrixTransformer:
    def __init__(self):
        pass

    def calculate_means(self, array4):
        """
        Calcula a média dos elementos de array4 em janelas deslizantes de 59 elementos.

        Args:
            array4 (list): Lista de inteiros (0 ou 1).

        Returns:
            list: Lista com a média dos elementos em janelas de 59 elementos.
        """
        array6 = []
        array7 = []
        for i in range(len(array4) - 1):
            array6.append(array4[i])
            if i >= 59:
                order = float(np.mean(array6[-59:]))
                array7.append(order)
        return array7

    def calculate_orders(self, array4):
        """
        Calcula a soma dos elementos de array4 em janelas deslizantes de 59 elementos.

        Args:
            array4 (list): Lista de inteiros (0 ou 1).

        Returns:
            list: Lista com a soma dos elementos em janelas de 59 elementos.
        """
        array5 = []
        for i in range(len(array4) - 1):
            if i >= 59:
                order = sum(array4[i - 59: i])
                array5.append(order)
        return array5

    def matriz(self, num_linhas, array):
        """
        Transforma um array unidimensional em uma matriz organizada por colunas.

        Args:
            array (list ou np.array): Lista de números a serem organizados.
            num_linhas (int): Número de linhas desejadas na matriz.

        Returns:
            np.array: Matriz ordenada.
        """
        t = len(array)
        if t % num_linhas != 0:
            raise ValueError("O tamanho do array deve ser múltiplo do número de linhas.")
        matriz = np.array(array).reshape(-1, num_linhas).T
        return matriz

    def tranforsmar_final_matriz(self, click, array1s, array1n):
        """
        Responsável por carregar a matriz final com múltiplas variáveis.

        Args:
            click (int): Valor que controla a segmentação das amostras.
            array1s (np.array): Array com entradas float (valores reais).
            array1n (np.array): Array com entradas int (valores binários 0/1).

        Returns:
            tuple: Matriz de features, matriz de saída e posição final.
        """
        n1 = len(array1n) - 61
        print(f'Inicial n1: {n1}')
        
        if n1 % click != 0:
            while n1 % click != 0:
                print(f'Ajustando: {(len(array1n) - 61)} % {click}')
                novo_array = np.random.choice([1, 0], size=60, p=[0.3, 0.7])
                array1n = np.concatenate((novo_array, array1n))
                n1 = len(array1n) - 61

        arrayacertos60 = self.calculate_orders(array1n)
        matrizacertos60 = self.matriz(click, arrayacertos60[1:])
        
        arraymediamovel = self.calculate_means(array1n)
        matrizmediamovel = self.matriz(click, arraymediamovel[1:])
        
        matrix1s, matrix1n = self.matriz(click, array1s[1:]), self.matriz(click, array1n[1:])
        matrix1s, matrix1n = matrix1s[:,1:], matrix1n[:,1:]

        print("Shapes antes do empilhamento:")
        print(matrix1n.shape, matrix1s.shape, matrizacertos60.shape, matrizmediamovel.shape)

        posicao0 = int((click // 60) - 1)

        # Empilhar todas as matrizes no eixo 2
        X_stack = np.stack([matrix1s, matrizacertos60, matrizmediamovel], axis=2)
        matrix1s = X_stack.reshape(60, -1)  # Shape final para entrada no modelo

        print("Shapes finais:")
        print(matrix1s.shape, matrix1n.shape)

        return matrix1s, matrix1n, posicao0
