import numpy as np

def transformar_em_matriz(array, num_linhas):
    """
    Transforma um array unidimensional em uma matriz organizada por colunas.
    
    Args:
        array (list ou np.array): Lista de números a serem organizados.
        num_linhas (int): Número de linhas desejadas na matriz.

    Returns:
        np.array: Matriz ordenada.
    """
    # Converte para numpy array para facilitar a manipulação
    array = np.array(array)
    
    # Calcula o número de colunas necessário
    num_colunas = len(array) // num_linhas
    
    # Reshape para matriz (por linha) e depois transpõe para organizar por colunas
    matriz = array.reshape(num_linhas, num_colunas).T
    
    return matriz  # Retorna como lista para melhor legibilidade

# Exemplo de uso
entrada = [1, 2, 3, 4, 5, 6, 7, 8, 9,10]
num_linhas = 2
resultado = transformar_em_matriz(entrada, num_linhas)

# Exibe o resultado
print(resultado)
