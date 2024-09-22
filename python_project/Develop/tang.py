import numpy as np

# Coleta de 120 entradas iniciais
i = 0
data_teste, array1 =  [], []

while i <= 120:
    print(24*'*-')
    print(f'Rodada: {i}')
    odd = input("Insira o número da odd: ").replace(",",".")
        
    if float(odd) >= 2:
        array1.append(1)
    else:
        array1.append(0)

    if i >= 60:
        array2 = array1[i - 60: i]
        print(len(array2))
        media = sum(array2)/60
        data_teste.append(media)
        print(f'Media60: {media}')
        array2 = []
    i += 1

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

# Função para rodar o modelo genético
def modelo(data_teste):
    # Parâmetros do algoritmo genético
    populacao_tamanho = 100
    geracoes = 1000
    taxa_mutacao = 0.001
    dados_reais = data_teste  # Dados pseudo-aleatórios iniciais

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
    return melhor_individuo

# Executa o modelo com os 120 dados coletados inicialmente
melhor_individuo = modelo(data_teste)
amplitude, frequencia, offset, ruido = melhor_individuo

# Gera as próximas 60 previsões
novas_entradas = gerar_oscillacao(amplitude, frequencia, offset, abs(ruido), 60)

# Gera os dados reais para comparação (pode ser real ou aleatório para teste)
dados_reais_novos = np.random.uniform(0.30, 0.60, 60)

# Comparação entre previsões e dados reais
comparacao = np.vstack((novas_entradas, dados_reais_novos)).T
print("Comparação das previsões com os dados reais (60 novas entradas):")
print("Previsões vs Reais")
for i in range(len(novas_entradas)):
    print(f"{novas_entradas[i]} vs {dados_reais_novos[i]}")

# Atualiza os dados reais para incluir as novas previsões
dados_reais = np.append(data_teste, dados_reais_novos)

# Agora pode continuar treinando com os novos dados atualizados
melhor_individuo_atualizado = modelo(dados_reais)
