import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import time

data1 = pd.read_csv('/home/darkcover/Documentos/Out/dados/DOUBLE - 17_09_s1.csv')

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
data_teste, array1, array3, array4, data_teste1, data_teste2 = [], [], [], [], [], []

while i <= 1280:
    print(24*'*-')
    print(f'Rodada: {i}')
    if i < 361:
        print(data1['Entrada'][i].replace(",","."))
        odd = data1['Entrada'][i].replace(",",".")
    else:
        odd = input("Insira o número da odd: ").replace(",",".")
        
    #Insere 0, 1 no array1, no qual é responsavel por guardar as entradas maiores do que 2.
    if float(odd) >= 2:
        array1.append(1)
    else:
        array1.append(0)

    #A partir da entrada 60, o array2 salva as últimas 60 entradas do array1 e em seguida calcula-se a média das últimas 60 entradas salvando-os em array chamado data_teste.
    if i >= 60:
        array2 = array1[i - 60: i]
        print(len(array2))
        
        media = sum(array2)/60
        data_teste.append(media)

        desvpad = np.std(array2, ddof=1) #desvio padrão amostral, importante lembrar
        data_teste1.append(desvpad)

        print(f'Media60: {media} \nDesvio Padrão60: {desvpad}')
        array2 = []

    if i >= 122:
        #últimas 60 entradas das médias
        array3 = data_teste[len(data_teste) - 60: len(data_teste)]
        #últimas 60 entradas do desvio padrão
        array4 = data_teste1[len(data_teste1) - 60: len(data_teste1)]
        
        # Calculando a correlação de Pearson
        correlacao, p_valor = pearsonr(array3, array4)

        print(f"Correlação de Pearson: {correlacao}")
        print(f"Valor-p: {p_valor}")

        data_teste2.append(correlacao)
        
        #print(len(array3), len(array4))
        
    if i % 60 == 0 and i >= 360:
        print(f'Executando o modelo após {i} entradas coletadas inicialmente:')
        # Executa o modelo com os 180 dados coletados inicialmente
        melhor_individuo = modelo(data_teste2)
        amplitude, frequencia, offset, ruido = melhor_individuo
        print("Melhor solução:", melhor_individuo)
        
        print("Gerando novas entradas, a partir das últimas entradas:")
        # Gerando as próximas 60 previsões
        novas_entradas = gerar_oscillacao(amplitude, frequencia, offset, abs(ruido), 60)
        ##Adicionar um gráfico nessa parte
        print(f'Entradas criadas: {len(novas_entradas)}')  # Deve imprimir 60
        j = 0

    if i >= 360 and j <= 58:
        if j >= 2:
            by_sinal = novas_entradas[j + 1] - novas_entradas[j]
        if by_sinal > 0 and j >= 1:
            print("Gráfico deve subir")
            print(f'Segundo a previsão: \nA média atual: {novas_entradas[j]} \nA média prevista: {novas_entradas[j + 1]}')
        elif by_sinal == 0 and j >= 1:
            print("Gráfico deve se manter o mesmo...")
            print(f'Segundo a previsão: \nA média atual: {novas_entradas[j]} \nA média prevista: {novas_entradas[j + 1]}')
        else:
            print('Gráfico deve descer')
            print(f'Segundo a previsão: \nA média atual: {novas_entradas[j]} \nA média prevista: {novas_entradas[j + 1]}')
        j += 1
    
    i += 1