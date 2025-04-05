import scipy.stats as stats

def calcular_distribuicao_binomial(array):
    # Tamanho do array e número de 1s (sucessos)
    n = len(array)
    num_sucessos = sum(array)
    
    # Estimativa da probabilidade de sucesso (média dos 1s no array)
    prob_sucesso = num_sucessos / n
    
    # Função de distribuição binomial
    # pmf calcula a probabilidade de termos exatamente 'num_sucessos' sucessos em 'n' tentativas
    probabilidade_binomial = stats.binom.pmf(num_sucessos, n, prob_sucesso)
    
    return probabilidade_binomial

array = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]  # Um exemplo de array com 0s e 1s
probabilidade_binomial = calcular_distribuicao_binomial(array)
print("Probabilidade binomial:", probabilidade_binomial)
