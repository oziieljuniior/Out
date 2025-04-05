def ponderar_lista(lista):
    """
    Função que realiza uma ponderação dos elementos de uma lista de 0 e 1.
    Os pesos aumentam linearmente conforme percorremos a lista.

    Args:
        lista (list): Lista de inteiros contendo apenas 0 e 1.

    Returns:
        int: Resultado ponderado, 0 ou 1.
    """
    n = len(lista)

    if n == 0:
        raise ValueError("A lista não pode estar vazia.")

    # Criar pesos crescentes
    pesos = [i + 1 for i in range(n)]

    # Calcular a soma ponderada
    soma_ponderada = sum(elemento * peso for elemento, peso in zip(lista, pesos))

    # Calcular o total de pesos
    total_pesos = sum(pesos)

    # Determinar o valor ponderado
    resultado = soma_ponderada / total_pesos

    # Retornar 1 se o valor ponderado for >= 0.5, senão 0
    return 1 if resultado >= 0.5 else 0

# Exemplo de uso
lista_exemplo = [0, 1, 0, 1, 1]
resultado = ponderar_lista(lista_exemplo)
print(f"Resultado ponderado: {resultado}")
