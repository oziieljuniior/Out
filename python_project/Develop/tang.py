import numpy as np

# Função para gerar oscilação limitada a uma variação máxima de ±0,02
def gerar_oscillacao(amplitude, frequencia, offset, ruido, tamanho, media_inicial=0.5):
    x_data = np.linspace(0, 1000, tamanho)
    osc = amplitude * np.sin(frequencia * x_data) + offset
    osc_ruido = osc + np.random.normal(0, abs(ruido), len(x_data))
    
    # Iniciando com a média inicial
    osc_final = [media_inicial]
    
    # Limitando variações a no máximo ±0,02 em relação ao valor anterior
    for i in range(1, len(osc_ruido)):
        proximo_valor = osc_final[-1] + np.clip(osc_ruido[i] - osc_final[-1], -0.02, 0.02)
        osc_final.append(proximo_valor)

    return np.array(osc_final)

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
    taxa_mutacao = 0.0015
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


# Coleta de 120 entradas iniciais
i, j, by_sinal = 0, 0, 0
data_teste, array1 =  [], []

while i <= 1280:
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

    if i % 60 == 0 and i >= 120:
        print(f'Executando o modelo após {i} entradas coletadas inicialmente:')
        # Executa o modelo com os 120 dados coletados inicialmente
        melhor_individuo = modelo(data_teste)
        amplitude, frequencia, offset, ruido = melhor_individuo
        print("Melhor solução:", melhor_individuo)
##Observação, aqui devo treinar com duas opções. A primeira é trabalhar com as novas entradas sendo atualizada
        print("Gerando novas entradas, a partir das últimas entradas:")
        # Gera as próximas 60 previsões
        novas_entradas = gerar_oscillacao(amplitude, frequencia, offset, abs(ruido), 60)
        print(f'Entradas criadas: {len(novas_entradas)}')
        j = 0

    if i >= 120 and j <= 58:
        if j >= 2:
            by_sinal = novas_entradas[j + 1] - novas_entradas[j]
        if by_sinal >= 0 and j >= 1:
            print("Gráfico deve subir")
            print(novas_entradas[j + 1])
        else:
            print('Gráfico deve descer')
            print(novas_entradas[j + 1])

        j += 1
    
    i += 1