def ponderar_lista_personalizada(lista):
    """
    Realiza uma ponderação dos elementos da lista, atribuindo o maior peso à primeira coluna,
    o segundo maior peso à última coluna, e pesos crescentes às colunas intermediárias.

    Args:
        lista (list): Lista de inteiros contendo apenas 0 e 1.

    Returns:
        int: Resultado ponderado, 0 ou 1.
    """
    n = len(lista)
    if n == 0:
        raise ValueError("A lista não pode estar vazia.")

    # Configurar pesos
    if n == 1:
        pesos = [1]  # Caso especial de lista com um único elemento
    else:
        maior_peso = n  # Maior peso para a primeira coluna
        segundo_maior_peso = n - 1  # Segundo maior peso para a última coluna
        
        # Pesos intermediários crescentes
        pesos_intermediarios = list(range(2, segundo_maior_peso + 1))
        
        # Compor a lista completa de pesos
        pesos = [maior_peso] + pesos_intermediarios + [segundo_maior_peso]

    # Calcular soma ponderada e total de pesos
    soma_ponderada = sum(elemento * peso for elemento, peso in zip(lista, pesos))
    total_pesos = sum(pesos)

    # Determinar o valor ponderado
    resultado = soma_ponderada / total_pesos

    # Retornar 1 se média ponderada >= 0.5, senão 0
    return 1 if resultado >= 0.5 else 0

# Exemplo de uso
lista_exemplo = [0, 1, 0, 1, 1]
resultado = ponderar_lista_personalizada(lista_exemplo)
print(f"Resultado ponderado: {resultado}")
