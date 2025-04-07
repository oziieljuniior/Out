import numpy as np

def ponderar_lista_avancada(lista, base=1.2):
    """
    Realiza uma ponderação dos elementos da lista com pesos exponenciais crescentes.

    Args:
        lista (list): Lista de inteiros contendo apenas 0 e 1.
        base (float): Base da função exponencial. Deve ser maior que 1.

    Returns:
        int: Resultado ponderado, 0 ou 1.
    """
    n = len(lista)
    if n == 0:
        raise ValueError("A lista não pode estar vazia.")
    
    # Calcular pesos exponenciais
    pesos = [base ** i for i in range(n)]
    
    # Calcular soma ponderada e total de pesos
    soma_ponderada = sum(elemento * peso for elemento, peso in zip(lista, pesos))
    total_pesos = sum(pesos)
    
    # Retornar 1 se média ponderada >= 0.5, senão 0
    return 1 if soma_ponderada / total_pesos >= 0.5 else 0

# Exemplo
lista_exemplo = [0, 1, 0, 1, 1]
resultado = ponderar_lista_avancada(lista_exemplo, base=1.5)
print(f"Resultado ponderado: {resultado}")
