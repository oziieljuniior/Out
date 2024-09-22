import numpy as np

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
    # Penalização para previsões com variações abruptas no ruído
    penalidade = np.mean(np.abs(np.diff(previsoes))) * 0.1  # Penaliza grandes mudanças
    return -erro - penalidade  # Fitness negativo porque queremos minimizar o erro

# Função de crossover
def crossover(pai1, pai2):
    return [(gene1 + gene2) / 2 for gene1, gene2 in zip(pai1, pai2)]

def mutacao(individuo, taxa_mutacao=0.01):
    # Aplica mutação em cada gene com uma pequena chance
    for i in range(len(individuo)):
        if np.random.rand() < taxa_mutacao:
            individuo[i] += np.random.normal(0, 0.1)  # Mutação mais agressiva
    return individuo

# Função para rodar o modelo genético
def modelo(data_teste):
    # Parâmetros do algoritmo genético
    populacao_tamanho = 200
    geracoes = 1000
    taxa_mutacao = 0.0015
    dados_reais = data_teste  # Dados pseudo-aleatórios iniciais

    # Inicializa uma população de indivíduos
    populacao = [np.random.uniform(0, 1, 4) for _ in range(populacao_tamanho)]  # Amplitude, freq, offset, ruído

    fitness_history = []

    for geracao in range(geracoes):
        fitness_scores = [fitness_function(individuo, dados_reais) for individuo in populacao]
        fitness_history.append(np.max(fitness_scores))  # Armazena o melhor fitness da geração
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
        print('Executando o modelo após 120 entradas coletadas inicialmente:')
        # Executa o modelo com os 120 dados coletados inicialmente
        melhor_individuo = modelo(data_teste)
        amplitude, frequencia, offset, ruido = melhor_individuo
        print("Melhor solução:", melhor_individuo)
        # Gerando novas entradas a partir das últimas entradas
        novas_entradas = gerar_oscillacao(amplitude, frequencia, offset, abs(ruido), 60)
        print(f'Entradas criadas: {len(novas_entradas)}')
        j = 0

    if i >= 120 and j <= 58:
        if j >= 2:
            by_sinal = novas_entradas[j + 1] - novas_entradas[j]
        if by_sinal >= 0 and j >= 1:
            print("Gráfico deve subir")
        else:
            print('Gráfico deve descer')
        j += 1
    
    i += 1
