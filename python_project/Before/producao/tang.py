import numpy as np
import pandas as pd

# Função para gerar oscilação com ruído
def gerar_oscillacao(amplitude, frequencia, offset, ruido, tamanho):
    x_data = np.linspace(0, 1000, tamanho)
    osc = amplitude * np.sin(frequencia * x_data) + offset
    # Garante que o ruído seja positivo
    osc_ruido = osc + np.random.normal(0, abs(ruido), len(x_data))
    return osc_ruido

# Função de fitness (erro médio absoluto)
def fitness_function(individuo, dados_reais):
    amplitude, frequencia, offset, ruido = individuo
    previsoes = gerar_oscillacao(amplitude, frequencia, offset, ruido, len(dados_reais))
    erro = np.mean(np.abs(previsoes - dados_reais))
    return -erro  # Fitness negativo porque queremos minimizar o erro

# Função de crossover
def crossover(pai1, pai2):
    return [(gene1 + gene2) / 2 for gene1, gene2 in zip(pai1, pai2)]

# Função de mutação
def mutacao(individuo, taxa_mutacao=0.01):
    return [gene + np.random.normal(0, taxa_mutacao) if np.random.rand() < 0.1 else gene for gene in individuo]

# Parâmetros do algoritmo genético
populacao_tamanho = 100
geracoes = 500
taxa_mutacao = 0.01
dados_reais = np.random.uniform(0.30, 0.60, 1000)  # Dados pseudo-aleatórios iniciais

# Inicializa uma população de indivíduos
populacao = [np.random.uniform(0, 1, 4) for _ in range(populacao_tamanho)]  # Amplitude, freq, offset, ruído

for geracao in range(geracoes):
    fitness_scores = [fitness_function(individuo, dados_reais) for individuo in populacao]
    sorted_population = [populacao[i] for i in np.argsort(fitness_scores)]
    populacao = sorted_population[-populacao_tamanho//2:]  # Elitismo
    nova_populacao = []
    for _ in range(populacao_tamanho // 2):
        idx_pai1, idx_pai2 = np.random.choice(len(populacao), 2, replace=False)
        pai1, pai2 = populacao[idx_pai1], populacao[idx_pai2]
        filho = crossover(pai1, pai2)
        filho = mutacao(filho, taxa_mutacao)
        nova_populacao.append(filho)
    populacao += nova_populacao

# Melhor solução encontrada
melhor_individuo = populacao[np.argmax(fitness_scores)]
amplitude, frequencia, offset, ruido = melhor_individuo
print("Melhor solução:", melhor_individuo)

# Gera as próximas 60 previsões
novas_entradas = gerar_oscillacao(amplitude, frequencia, offset, abs(ruido), 60)

# Gera os dados reais para comparação (dados pseudo-aleatórios)
dados_reais_novos = np.random.uniform(0.30, 0.60, 60)

# Comparação entre previsões e dados reais
comparacao = np.vstack((novas_entradas, dados_reais_novos)).T

# Exibe a comparação
print("Comparação das previsões com os dados reais (60 novas entradas):")
print("Previsões vs Reais")
for i in range(len(novas_entradas)):
    print(f"{novas_entradas[i]} vs {dados_reais_novos[i]}")

# Atualiza os dados reais para incluir as novas previsões
dados_reais = np.append(dados_reais, dados_reais_novos)
